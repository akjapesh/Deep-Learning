import argparse
import os
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow.compat.v1 as tf

from tensorflow.keras.layers import Input, Lambda, Conv2D
from tensorflow.keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from tensorflow.python.framework import ops
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.keras.backend as k


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.disable_eager_execution()

def yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold=0.6):
    box_scores=box_confidence*box_class_probs

    box_classes=k.argmax(box_scores,axis=-1)
    box_class_scores=k.max(box_scores,axis=-1)

    filtering_mask=box_class_scores>=threshold
    scores=tf.boolean_mask(box_class_scores,filtering_mask)
    boxes=tf.boolean_mask(boxes,filtering_mask)
    classes=tf.boolean_mask(box_classes,filtering_mask)

    return scores,boxes,classes


def iou(box1,box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    xi1 = max(box1_x1,box2_x1)
    yi1 = max(box1_y1,box2_y1)
    xi2 = min(box1_x2,box2_x2)
    yi2 = min(box1_y2,box2_y2)
    inter_width = max(xi2-xi1,0)
    inter_height = max(yi2-yi1,0)
    inter_area = inter_height*inter_width
    ### END CODE HERE ###    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    union_area = box2_area+box1_area-inter_area
    ### END CODE HERE ###

    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area/union_area
    ### END CODE HERE ###

    return iou

def yolo_non_max_suppression(scores,boxes,classes,max_boxes=10,iou_threshold=0.5):
    max_boxes_tensor = tf.keras.backend.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    tf.keras.backend.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ### START CODE HERE ### (≈ 1 line)
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold)
    ### END CODE HERE ###

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = k.gather(scores,nms_indices)
    boxes = k.gather(boxes,nms_indices)
    classes = k.gather(classes,nms_indices)

    return scores,boxes,classes

def yolo_eval(yolo_outputs,image_shape=(720.,1280.),max_boxes=10,score_threshold=0.6,iou_threshold=0.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores,boxes,classes,max_boxes,iou_threshold)

    ### END CODE HERE ###

    return scores, boxes, classes

sess=tf.keras.backend.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)
yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def predict(sess,image_file):
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run(fetches=[scores,boxes,classes],
       feed_dict={yolo_model.input: image_data,
                  tf.keras.backend.learning_phase():0
       })
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    plt.imshow(output_image)

    return out_scores, out_boxes, out_classes
out_scores, out_boxes, out_classes = predict(sess, "test.jpg")