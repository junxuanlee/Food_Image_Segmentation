from config import *
from segmentation_models import Unet
from tensorflow.keras.models import Model

#================================================================================================================================================================================
# UNET MODEL & CONFIGURATIONS
#================================================================================================================================================================================
backbone = 'resnet152'
shape=(128, 128, 3)
weights_pretrained = 'imagenet'
   
activ = 'relu'
freeze = True #If True, train only randomly initialized decoder in order not to damage weights of properly trained encoder

unet0 = Unet(backbone, input_shape=shape, encoder_weights=weights_pretrained,classes=n_classes, activation=activ, encoder_freeze=freeze) 
layer_name = 'final_conv'
unet= Model(inputs=unet0.input, outputs=unet0.get_layer(layer_name).output)
#================================================================================================================================================================================
