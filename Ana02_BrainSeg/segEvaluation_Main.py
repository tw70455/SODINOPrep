import numpy as np
import sys,os,glob
sys.path.append("..")
import segEvaluation_Func as eva
from paraClass import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

organ = 'RatBrain'
stage = 'val'


# Automatically get the current script folder path
base_dir = os.path.dirname(os.path.abspath(__file__))


#imgpath = '/Volumes/ProcessDisk/ZTE/PreprocessingCode/Ana02_BrainSeg/IMG' ## Resampled Intensity Image Folder
# Define paths relative to the script location
imgpath = os.path.join(base_dir, 'IMG')             # Resampled Intensity Image Folder
labelpath = '' ## Resampled Label Mask Folder
#savepath = '/Volumes/ProcessDisk/ZTE/PreprocessingCode/Ana02_BrainSeg/IMGMask' ## Result Save Folder
savepath = os.path.join(base_dir, 'IMGMask')         # Result Save Folder

"""Not change the following parameters"""
normparas = NormParas()
normparas.method = "minmax"

preparas = PreParas()
preparas.patchdims = [64,64,64]
preparas.patchlabeldims = [64,64,64]
preparas.patchstrides = [16,16,16]

preparas.organname = organ
preparas.stage = stage
preparas.nclass = 2
preparas.ndim = '3D_LabelHot'
preparas.issubtract = 0

kerasparas = KerasParas()
kerasparas.outID = 0
kerasparas.thd = 0.5
kerasparas.loss = 'dice_coef_loss'
organids = [1]

kerasparas.imgformat = 'channels_last'
kerasparas.modelname = '3D_Unet'
"""------------------------------------"""

#kerasparas.modelpath = '/Users/liming/Dropbox/Brain/Rat/Preprocessing/Ana02_BrainSeg/Evaluation/Rat_Brain-3D_Unet-72-0.9723-0.9716.hdf5' ## Path to model
kerasparas.modelpath = os.path.join(base_dir, 'SORDINO_UNet_model.hdf5') # Trained model file path

# eva.online_seg_evaluation(imgpath,labelpath,savepath,preparas,normparas,organids,kerasparas)
eva.online_seg_prediction(imgpath,savepath,preparas,normparas,organids,kerasparas)
