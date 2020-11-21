import matplotlib
import matplotlib.pyplot as plt

import os
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

pipeline_config = '<path-to-model>/pipeline.config'
model_dir = '<path-to-model>/checkpoint'
image_dir = '<path-to-image-dir>'
output_dir = '<path-to-output-dir>'

# Load model

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config,
                                      is_training=False)

# Restore checkpoint

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()


# Detection function

def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


detect_fn = get_model_detection_function(detection_model)

# Label maps

label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)

categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)


# Test images

def get_np_array_from_file(filepath):
    img_data = tf.io.gfile.GFile(filepath, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (width, height) = image.size
    return np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)


images_np = []
images_np_with_detections = []

detections_list = []
predictions_dict_list = []
shapes_list = []

for image_path in os.listdir(image_dir):
    if image_path.endswith('.xml'):
        continue

    image_path = os.path.join(image_dir, image_path)
    image_np = get_np_array_from_file(image_path)
    images_np.append(image_np)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    detections_list.append(detections)
    predictions_dict_list.append(predictions_dict)
    shapes_list.append(shapes)
    images_np_with_detections.append(image_np.copy())

label_id_offset = 1
i = 1

for image_np_with_detections in images_np_with_detections:
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections_list[i-1]['detection_boxes'][0].numpy(),
        (detections_list[i-1]['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections_list[i-1]['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=None,
        keypoint_scores=None,
        keypoint_edges=None)

    img = Image.fromarray(image_np_with_detections, 'RGB')
    # untested, possible compatibility issues with windows in os.path.join
    img.save(os.path.join(output_dir, (str(i) + '.jpg')))
    #print(detections_list[0])
    i += 1
