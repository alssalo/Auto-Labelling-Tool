# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pathlib
from PIL import Image

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

def run_detection(image_path,model_value,threshold,items):
    #print(image_path,"from")
    print(items,"model")

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = model_value

    CWD_PATH = "C:/Users/Jo's/Documents/object detection/models/research/object_detection"

    #Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('test_images')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))


    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    #print(str(TEST_IMAGE_PATHS[0]),"here")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)
    
    indices = np.array([[0,]])
    
    cat_list = [] 
    for i in list(category_index.values()):
        cat_list.append(list(i.values()))

    selected_classes= []
    for j in items:
        for i in cat_list:
            if i[1]==j:
                selected_classes.append(i[0])
    print(selected_classes,"classes")
    for i in selected_classes:
           indices = np.concatenate((indices, (np.argwhere(classes == i))))
        
    indices = np.delete(indices, 0,axis=0)
    indices = np.sort(indices, axis=0)
    boxes = np.squeeze(boxes[indices])
    scores = np.squeeze(scores[indices])
    classes = np.squeeze(classes[indices])

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes.astype(np.int32),
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=threshold)
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    im1 = im_pil.save("sample.jpg") 
    #im_pil.show()
    return(im1)

    # All the results have been drawn on image. Now display the image.
    #cv2.imshow('Object detector', image)

    # Press any key to close the image
    #cv2.waitKey(0)
    
    # Clean up
    #cv2.destroyAllWindows()

    
