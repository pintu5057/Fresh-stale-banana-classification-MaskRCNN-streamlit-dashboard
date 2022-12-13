import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
import streamlit as st

# Root directory of the project
ROOT_DIR = "D:\\MaskRCNN-2021\\Mask-RCNN-on-Custom-Dataset"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.model import MaskRCNN

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + two banana classes# change for number of classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset, json_path):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "FB") # add your classes
        self.add_class("object", 2, "SB") # add your classes
     

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(json_path))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"FB": 1, "SB": 2} # Add the classes accordingly

            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
     
            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



########################Inference with pre-trained model ####################
img = cv2.imread(r'D:/Fruit_Veggies_detection/Sample_data/val/rotated_by_15_Screen Shot 2018-06-12 at 9.45.22 PM.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Conver BGR to RGB
# define the test configuration
 
# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config = CustomConfig())
# load trained model weights
rcnn.load_weights('Banana_test2.h5', by_name=True)


# import pickle

# with open('model.pickle','wb') as f:
#     pickle.dump(rcnn,f)


results = rcnn.detect([img], verbose=0)


# define two classes that the Banana test model knowns about
class_names = ['BG','FB', 'SB' ]
r = results[0]
max_bbox=[]

for b in range(len(r['rois'])):
    (startY, startX, endY, endX) = r["rois"][b]
    (W,H) = (endX - startX, endY-startY)
    max_bbox.append(W+H)
max_idx = np.argmax(max_bbox)

r_updated = dict()

r_updated['class_ids'] = r['class_ids'][[max_idx]]
m = r['masks'][:,:,max_idx]
mm = np.reshape(m,[np.shape(m)[0],-1,1])
r_updated['masks'] = mm
r_updated['rois'] = r['rois'][[max_idx]]
r_updated['scores'] = r['scores'][[max_idx]]

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Conver RGB to BGR

# show photo with bounding boxes, masks, class labels and scores
#display_instances(img, r_updated['rois'], r_updated['masks'], r_updated['class_ids'], class_names, r_updated['scores'], show_mask=True, show_mask_polygon=False)

r = r_updated

color1 = [255,0,100]
color2 = [0,0,255]
for i in range(0, len(r["scores"])):
    (startY, startX, endY, endX) = r["rois"][i]    
    
    classID = r["class_ids"][i]
    label = ['BG','Fresh Banana','Stale Banana'] # define two classes that the Banana test model knowns about + Background
    score = r["scores"][i]
 	#color = [int(c) for c in np.array(COLORS[classID]) * 255]
 	# draw the bounding box, class label, and score of the object
    cv2.rectangle(img, (startX, startY), (endX, endY), color1, 2)
    text = "{}:{: .3f}".format(label[classID],score)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, color2, 2)
# show the output image


cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", img)
cv2.waitKey()
