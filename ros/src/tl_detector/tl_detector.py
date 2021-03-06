#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import numpy as np
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.light_classifier = TLClassifier('ssd_mobilenet_v1_coco_2017_11_17')
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def flatground_dist(self, coord1, coord2):
        x1,y1 = coord1
        x2,y2 = coord2
        return ((x1-x2)**2 + (y1-y2)**2)**0.5

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        if state == TrafficLight.RED:
            rospy.loginfo("RED LIGHT DETECTED")
        elif state == TrafficLight.GREEN:
            rospy.loginfo("GREEN LIGHT DETECTED")
        else:
            rospy.loginfo("NO LIGHT DETECTED")

        statedict = {TrafficLight.UNKNOWN:"UNKNOWN", TrafficLight.RED: "RED", TrafficLight.GREEN:"GREEN", TrafficLight.YELLOW:"YELLOW"}
        # rospy.loginfo("LIGHT STATE: " + str(statedict[state]))
        # rospy.loginfo("LIGHT WP: " + str(light_wp))
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            # rospy.loginfo("state count exceeds threshold: " + str(light_wp))
            self.upcoming_red_light_pub.publish(Int32(light_wp))

        else:
            # rospy.loginfo("else: " + str(self.last_wp))
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            rospy.loginfo("IMAGE WRITE ERROR" )
        else:
            # Save your OpenCV2 image as a jpeg 
            ts = rospy.get_rostime()
            # light_dist, state = self.get_closest_light_truth()


    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val>0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return  closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # return light.state

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose and self.base_waypoints and self.waypoint_tree):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)    
            diff = len(self.waypoints.waypoints)
            min_dist = float("inf")
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                light_wp_idx = self.get_closest_waypoint(line[0], line[1])
                dist = self.distance(self.waypoints.waypoints[car_wp_idx].pose.pose.position.x, 
                                        self.waypoints.waypoints[car_wp_idx].pose.pose.position.y, 
                                        self.waypoints.waypoints[light_wp_idx].pose.pose.position.x, 
                                        self.waypoints.waypoints[light_wp_idx].pose.pose.position.y)

                if dist < min_dist and light_wp_idx > car_wp_idx:
                    min_dist = dist
                    closest_light = light
                    line_wp_idx = light_wp_idx

        if closest_light:
            state = self.get_light_state(light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

    def distance(self, x1, y1, x2, y2):
        return ((x1-x2)**2 + (y1-y2)**2)**0.5

if __name__ == "__main__":
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
