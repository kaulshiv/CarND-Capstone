from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import sys
import tarfile
import os
from PIL import Image

# from object_detection.utils import ops as utils_ops


# patch tf1 into `utils.ops`
# utils_ops.tf = tf.compat.v1

# Patch the location of gfile
# tf.gfile = tf.io.gfile

class TLClassifier(object):
    def __init__(self, model_name):
        #TODO load classifier
        self.model_name = model_name
        self.model = None
        self.input_image = None
        self.light_prediction = None
        self.counter = 0

        self.load_model()

    def load_model(self):
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = self.model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=self.model_name, 
            origin=base_url + model_file,
            untar=True)

        model_dir = os.path.join(model_dir, "saved_model")
        self.model = tf.saved_model.load(str(model_dir))

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        self.input_image = image
        if self.detect_and_classify():
            return self.light_prediction

        return TrafficLight.UNKNOWN

    def detect_and_classify(self):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        
        # Actual detection.
        output_dict = self.run_inference_for_single_image(self.input_image)

        final_img = None
        light_detected = False
        light_prediction = None
        for i, classidx in enumerate(output_dict['detection_classes']):
            if classidx==10:
                light_detected = True
                cropped_img = self.get_crop(output_dict['detection_boxes'][i, :])
                break

        if light_detected:
            cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            light_prediction = self.classify_light()
            return True
            
        return False

    def run_inference_for_single_image(self, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        
        # # Handle models with masks:
        # if 'detection_masks' in output_dict:
        #     # Reframe the the bbox mask to the image size.
        #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        #             output_dict['detection_masks'], output_dict['detection_boxes'],
        #             image.shape[0], image.shape[1])      
        #     detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
        #                                     tf.uint8)
        #     output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
            
        return output_dict

    def classify_light(self):
        img_hsv=cv2.cvtColor(self.input_image, cv2.COLOR_BGR2HSV)
        img_rgb=cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
        mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))

        ## Merge the mask and crop the red regions
        red_mask = cv2.bitwise_or(mask1, mask2)
        target_red = cv2.bitwise_and(self.input_image, self.input_image, mask=red_mask)

        ## mask of green (36,0,0) ~ (70, 255,255)
        green_mask = cv2.inRange(img_hsv, (36, 0, 0), (70, 255,255))
        target_green = cv2.bitwise_and(self.input_image, self.input_image, mask=green_mask)

        # join my masks
        num_red_pixels = np.sum(red_mask)
        num_green_pixels = np.sum(green_mask)

        if(num_red_pixels<num_green_pixels and num_green_pixels>self.input_image.size*0.2):
            self.light_prediction = TrafficLight.GREEN
        self.light_prediction = TrafficLight.RED

        final_img = Image.fromarray(img_rgb)
        final_img.save(str(self.light_prediction)+"_" + str(self.counter))
        self.counter+=1

    def get_crop(self, bbox):
        h, w, _ = self.input_image.shape
        bot, left, top, right = bbox
        bot = int(bot*h)
        top = int(top*h)
        left = int(left*w)
        right = int(right*w)
        return self.input_image[ bot:top, left:right, :]

