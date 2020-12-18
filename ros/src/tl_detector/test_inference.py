import numpy as np
import os
import os.path
import glob
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import time

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = os.path.join(model_dir, "saved_model")

  model = tf.saved_model.load(str(model_dir))

  return model


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
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
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def show_inference(model, light_classification, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)

    # Visualization of the results of a detection.
    #vis_util.visualize_boxes_and_labels_on_image_array(
    #    image_np,
    #    output_dict['detection_boxes'],
    #    output_dict['detection_classes'],
    #    output_dict['detection_scores'],
    #    category_index,
    #    instance_masks=output_dict.get('detection_masks_reframed', None),
    #    use_normalized_coordinates=True,
    #    line_thickness=8)

    final_img = None
    light_detected = False
    light_prediction = None
    for i, classidx in enumerate(output_dict['detection_classes'][0:3]):
        if classidx==10:
        #   print("det boxes >>>>> ", output_dict['boxes'][i, :], ", det score>>>>", output_dict['detection_scores']) 
            light_detected = True
            cropped_img = get_crop(image_np, output_dict['detection_boxes'][i, :])
            break

    if light_detected:
        cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        light_prediction = classify_light(cropped_img, image_path)
        
        final_img = Image.fromarray(cropped_gray) 
        final_img.save(os.path.join('predictions', light_classification + "_" + light_prediction + "_" +  image_path.split('/')[-1]))



#   final_img.save(os.path.join('outimgs', light_classification, image_path.split('/')[-1]))
    return final_img, light_detected, light_prediction



def get_crop(image, bbox):
    h, w, _ = image.shape
    bot, left, top, right = bbox
    bot = int(bot*h)
    top = int(top*h)
    left = int(left*w)
    right = int(right*w)
    return image[ bot:top, left:right, :]

def classify_light(image, image_path):
    img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    maskr0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    maskr1 = cv2.inRange(img_hsv, lower_red, upper_red)

    ## mask of green (36,0,0) ~ (70, 255,255)
    green_mask = cv2.inRange(img_hsv, (36, 0, 0), (70, 255,255))

    # join my masks
    red_mask = cv2.bitwise_or(maskr0,maskr1)
    num_red_pixels = np.sum(red_mask)
    num_green_pixels = np.sum(green_mask)
    target_red = cv2.bitwise_and(img, img, mask=red_mask)
    target_green = cv2.bitwise_and(img, img, mask=green_mask)

    final_img = Image.fromarray(target_red) 
    final_img.save(os.path.join( 'redmask',  image_path.split('/')[-1]))
    final_img = Image.fromarray(target_green) 
    final_img.save(os.path.join( 'greenmask',  image_path.split('/')[-1]))

    if(num_red_pixels<num_green_pixels):
        return "green"
    return "red"

if __name__=="__main__":
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    #print("category index >>>>> ", category_index)

    ptid = ['tflights/red', 'tflights/green', 'tflights/yellow', 'tflights/nolight']

    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    detection_model = load_model(model_name)

    num_lights, num_nolights = 0, 0
    false_positives, false_negatives = 0, 0

    correct_reds, correct_greens = 0, 0
    num_reds, num_greens = 0, 0

    for PATH_TO_TEST_IMAGES_DIR in ptid:
        TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR,"*.jpeg"))
        light_ground_truth = PATH_TO_TEST_IMAGES_DIR.split('/')[-1]
        
        for image_path in TEST_IMAGE_PATHS:
            t0 = time.time()
            final_img, light_detected, prediction = show_inference(detection_model, light_ground_truth, image_path)
            inf_time = time.time()-t0

            if light_ground_truth=="red":
                if prediction == light_ground_truth:
                    correct_reds += 1
                num_reds +=1
            elif light_ground_truth=="green":
                if prediction == light_ground_truth:
                    correct_greens += 1
                num_greens +=1


            # if light_ground_truth=="nolight":
            #     if light_detected:
            #         false_positives += 1
            #     num_nolights += 1
            # else:
            #     if not light_detected:
            #         false_negatives += 1
            #     num_lights += 1
            

            # print('image_path >>>>>>>', image_path, ", inf time: ", inf_time)

    # print('false positives >>>>>>>', false_positives)
    # print('num lights >>>>>>>', num_lights)
    # print('false negatives >>>>>>>', false_negatives)
    # print('num nolights >>>>>>>', num_nolights)
    print("correctly predicted reds: " + str(correct_reds) + "/" + str(num_reds))
    print("correctly predicted greens: " + str(correct_greens) + "/" + str(num_greens))
