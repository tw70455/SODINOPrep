from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D

# from keras.layers.normalization import BatchNormalization #original
from tensorflow.keras.layers import BatchNormalization

#from keras.layers.core import SpatialDropout3D, Activation #original
from tensorflow.keras.layers import SpatialDropout3D, Activation


from keras import backend as K

# from keras.layers.merge import concatenate #original
from tensorflow.keras.layers import concatenate

import math
import numpy as np

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
