import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
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
    
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"FB": 1, "SB": 2} # Add the classes accordingly

     
            num_ids = [name_dict[a] for a in objects]
     
   
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


        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    train_json_path = r'D:\Fruit_Veggies_detection\Sample_data\train\train_json.json'
    dataset_train = CustomDataset()
    dataset_train.load_custom(r"D:\Fruit_Veggies_detection\Sample_data", "train", train_json_path)
    dataset_train.prepare()

    # Validation dataset
    val_json_path = r'D:\Fruit_Veggies_detection\Sample_data\val\val_json.json'
    dataset_val = CustomDataset()
    dataset_val.load_custom(r"D:\Fruit_Veggies_detection\Sample_data", "val", val_json_path)
    dataset_val.prepare()
    


    
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')
			
				
				
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)


weights_path = COCO_WEIGHTS_PATH
        # Download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])


train(model)			

model_path = 'Banana_test2.h5'
model.keras_model.save_weights(model_path)




########################Inference with pre-trained model ####################
img = load_img(r'D:/Fruit_Veggies_detection/Sample_data/val/rotated_by_15_Screen Shot 2018-06-12 at 9.45.22 PM.png')
img = img_to_array(img)

# define the test configuration
 
# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config = CustomConfig())
# load trained model weights
rcnn.load_weights('Banana_test2.h5', by_name=True)

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


# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r_updated['rois'], r_updated['masks'], r_updated['class_ids'], class_names, r_updated['scores'], show_mask=True, show_mask_polygon=False)

display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], show_mask=True, show_mask_polygon=False)

# Performance evaluation



val_json_path = r'D:\Fruit_Veggies_detection\Sample_data\val\val_json.json'
dataset = CustomDataset()
dataset.load_custom(r"D:\Fruit_Veggies_detection\Sample_data", "val", val_json_path)
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

config = CustomConfig()
total_gt = np.array([]) 
total_pred = np.array([]) 
mAP_ = [] #mAP list

# Mean Average Precision calculation on "val" database

for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =modellib.load_image_gt(dataset, config, image_id)#, #use_mini_mask=False)

    # Run the model
    results = rcnn.detect([image], verbose=0)
    r = results[0]

    #precision_, recall_, AP_ 
    AP_, precision_, recall_, overlap_ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
 
    mAP_.append(AP_)
    print("Average precision of this image : ",AP_)
print("The actual mean average precision for the whole image set", sum(mAP_)/len(mAP_))



