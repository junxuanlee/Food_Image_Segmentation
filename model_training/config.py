#============================
#USE THIS WHEN USING GPU
#============================
#import tensorflow as tf
#tf.keras.backend.clear_session()
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

#============================
#USE THIS WHEN USING CPU
#============================

import tensorflow as tf
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
        print('asserted')
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import os
import cv2
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import imageio
import random
import albumentations as A
import tensorflow as tf


#================================================================================================================================================================================
# CHANGE ALL THE DIRECTORIES TO MATCH YOURS
#================================================================================================================================================================================
p1 = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/UNIMIB2016 Food Database/train/food_images/images/'
p2 = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/UNIMIB2016 Food Database/train/food_masks/images/'
p3 = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/UNIMIB2016 Food Database/val/food_images/images/'
p4 = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/UNIMIB2016 Food Database/val/food_masks/images/'

#Image and mask used inspect the model's training quality
image_path = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/UNIMIB2016 Food Database/test/food_images/images/20151130_122225.jpg'
food_mask_path = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/UNIMIB2016 Food Database/test/food_masks/images/20151130_122225.png'

image_path2 = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/python/test_image.png'

#Paths where all training results are stored
training_progress_path = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/training/training_progress/'
tensorboard_path = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/training/tensorboard_logs/'
checkpoint_filepath = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/ok/training/checkpoints/'

#================================================================================================================================================================================
# INPUT CONFIGURATIONS
#================================================================================================================================================================================
TARGET_SIZE = (128,128)
BATCH_SIZE = 6

classes = 12 #dont touch unless changed dataset with different classes
num_classes_to_keep = 0 #put '0' to keep all classes
num_filest = 722 #Number of training files
num_filesv = 192 #Number of validation files

if num_classes_to_keep == 0: #meaning all the classes are kept
    n_classes = classes + 1
else:
    n_classes = num_classes_to_keep + 1

#================================================================================================================================================================================
