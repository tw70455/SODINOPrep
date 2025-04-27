from keras import backend as keras
from functools import partial, update_wrapper

#from keras.utils.generic_utils import CustomObjectScope ## original
from tensorflow.keras.utils import CustomObjectScope


# from keras.applications.mobilenet import DepthwiseConv2D, relu6
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import os, h5py, sys, glob
import SimpleITK as sitk
import numpy as np
import scipy.io as sio
from PIL import Image
import skimage.morphology as skimor

Image.MAX_IMAGE_PIXELS = 1000000000
sys.dont_write_bytecode = True
sys.path.append("..")
sys.path.append("../..")

from paraClass import *
from diceLoss import *
import segMetrics as met

import time # liming


def min_max_normalization(img):
    newimg = img.copy()
    newimg = newimg.astype(np.float32)

    minval = np.min(newimg)
    maxval = np.max(newimg)
    newimg =(np.asarray(newimg).astype(np.float32) - minval)/(maxval-minval)
    return newimg

def label_adjustment(label,nclass,negids=[],sortids=[]):
    for i in negids:
        label[label==i] = 0
    if nclass==2:
        label[label!=0] = 1
    if len(sortids)!=0:
        labelcopy = label.copy()
        tab = 1
        for i in sortids:
            label[labelcopy==i] = tab
            tab +=1
    return label

def dim_2_categorical(label,numclass):
    dims = label.ndim
    if dims==2:
        col, row = label.shape
        exlabel = np.zeros((numclass, col, row))
        for i in range(0,numclass):
            exlabel[i,] = np.asarray(label == i).astype(np.uint8)
    elif dims==3:
        leng,col,row = label.shape
        exlabel  = np.zeros((numclass,leng,col,row))
        for i in range(0,numclass):
            exlabel[i,] = np.asarray(label == i).astype(np.uint8)
    return exlabel

"""--------------------------------------------------output labelmap--------------------------------------------"""
def out_LabelHot_map_2D(img,segout_LabelHot_map_3Dnet,preparas,kerasparas):
    # original out_LabelHot_map_2D
    # reset the variables
    patchdims = preparas.patchdims
    labeldims = preparas.patchlabeldims
    strides = preparas.patchstrides
    nclass = preparas.nclass

    # build new variables for output
    leng,col,row = img.shape
    categoricalmap = np.zeros((nclass, leng, col, row), dtype=np.uint8)
    likelihoodmap = np.zeros((leng, col, row), dtype=np.float32)
    countermap = np.zeros((leng,col,row), dtype=np.float32)
    lengstep = int(patchdims[0]/2)

    """-----predict the whole image from two directions, small to large and large to small----"""
    for i in range(0,leng-patchdims[0]+1,strides[0]):
        for j in range(0,col-patchdims[1]+1,strides[1]):
            for k in range(0,row-patchdims[2]+1,strides[2]):

                curpatch=img[i:i+patchdims[0],j:j+patchdims[1],k:k+patchdims[2]][:].reshape([1,patchdims[0],patchdims[1],patchdims[2]])
                if kerasparas.imgformat == 'channels_last':
                    curpatch = np.transpose(curpatch, (0, 2, 3, 1))

                curpatchoutput = segnet.predict(curpatch, batch_size=1, verbose=0)

                # if there are multiple outputs
                if isinstance(curpatchoutput,list):
                    curpatchoutput = curpatchoutput[kerasparas.outID]
                curpatchoutput = np.squeeze(curpatchoutput)

                curpatchoutlabel = curpatchoutput.copy()
                curpatchoutlabel[curpatchoutlabel>=kerasparas.thd] = 1
                curpatchoutlabel[curpatchoutlabel<kerasparas.thd] = 0

                middle = i + lengstep
                curpatchoutlabel = dim_2_categorical(curpatchoutlabel,nclass)

                categoricalmap[:, middle, j:j + labeldims[1], k:k + labeldims[2]] = categoricalmap[:, middle, j:j + labeldims[1], k:k + labeldims[2]] + curpatchoutlabel
                likelihoodmap[middle, j:j + labeldims[1], k:k + labeldims[2]] = likelihoodmap[middle, j:j + labeldims[1], k:k + labeldims[2]] + curpatchoutput
                countermap[middle, j:j + labeldims[1], k:k + labeldims[2]] += 1

    for i in range(leng, patchdims[0]-1,-strides[0]):
        for j in range(col, patchdims[1]-1,-strides[1]):
            for k in range(row, patchdims[2]-1,-strides[2]):

                curpatch=img[i-patchdims[0]:i,j-patchdims[1]:j,k-patchdims[2]:k][:].reshape([1,patchdims[0],patchdims[1],patchdims[2]])
                if kerasparas.imgformat == 'channels_last':
                    curpatch = np.transpose(curpatch, (0, 2, 3, 1))

                curpatchoutput = segnet.predict(curpatch, batch_size=1, verbose=0)

                if isinstance(curpatchoutput,list):
                    curpatchoutput = curpatchoutput[kerasparas.outID]
                curpatchoutput = np.squeeze(curpatchoutput)

                curpatchoutlabel = curpatchoutput.copy()
                curpatchoutlabel[curpatchoutlabel>=kerasparas.thd] = 1
                curpatchoutlabel[curpatchoutlabel<kerasparas.thd] = 0

                middle = i - patchdims[0] + lengstep
                curpatchoutlabel = dim_2_categorical(curpatchoutlabel,nclass)
                categoricalmap[:, middle, j-labeldims[1]:j, k-labeldims[2]:k] = categoricalmap[:, middle, j-labeldims[1]:j, k-labeldims[2]:k] + curpatchoutlabel
                likelihoodmap[middle, j-labeldims[1]:j, k-labeldims[2]:k] = likelihoodmap[middle, j-labeldims[1]:j, k-labeldims[2]:k] + curpatchoutput
                countermap[middle, j-labeldims[1]:j, k-labeldims[2]:k] +=1

    #####--------------------------------------------------------
    labelmap = np.zeros([leng,col,row],dtype=np.uint8)
    for idx in range(0,leng):
        curslicelabel = np.squeeze(categoricalmap[:, idx,].argmax(axis=0))
        labelmap[idx,] = curslicelabel

    countermap = np.maximum(countermap, 10e-10)
    likelihoodmap = np.divide(likelihoodmap,countermap)

    return labelmap,likelihoodmap


def out_LabelHot_map_3D(img,segnet,preparas,kerasparas,addinputlist=[]):
    # reset the variables
    patchdims = preparas.patchdims
    labeldims = preparas.patchlabeldims
    strides = preparas.patchstrides
    nclass = preparas.nclass
    meanvalue = preparas.meanvalue

    if meanvalue is None and preparas.issubtract:
        meanvalue = DB.mean_patch_generation(img,patchdims,3)
    if meanvalue is not None:
        meanvalue = meanvalue[np.newaxis,:]

    # build new variables for output
    leng,col,row = img.shape
    categoricalmap = np.zeros((nclass,leng,col,row), dtype=np.uint8)
    likelihoodmap = np.zeros((leng,col,row), dtype=np.float32)
    countermap = np.zeros((leng,col,row), dtype=np.float32)

    addinputnum = len(addinputlist)
    """-----predict the whole image from two directions, small to large and large to small----"""
    for i in range(0,leng-patchdims[0]+1,strides[0]):
        for j in range(0,col-patchdims[1]+1,strides[1]):
            for k in range(0,row-patchdims[2]+1,strides[2]):

                curpatch=img[i:i+patchdims[0],j:j+patchdims[1],k:k+patchdims[2]][:].reshape([1,1,patchdims[0],patchdims[1],patchdims[2]])
                for addidx in range(addinputnum):
                    curaddpatch = addinputlist[addidx][i:i+patchdims[0],j:j+patchdims[1],k:k+patchdims[2]][:].reshape([1,1,patchdims[0],patchdims[1],patchdims[2]])
                    curpatch = np.append(curpatch,curaddpatch,axis=1)

                if preparas.issubtract:
                    if meanvalue.shape!=curpatch.shape:
                        curpatch[:,0:meanvalue.shape[1],:,:,:] = curpatch[:,0:meanvalue.shape[1],:,:,:] - meanvalue
                    else:
                        curpatch = curpatch - meanvalue

                if kerasparas.imgformat == 'channels_last':
                    curpatch = np.transpose(curpatch,(0,2,3,4,1))

                if 'W_' in kerasparas.modelname:
                    curweight = np.ones(curpatch.shape,dtype=np.uint8) # shuai
                    # curweight = np.ones((curpatch.shape[:4]+(1,)), dtype=np.uint8)  # shuai for breast synseg
                    curpatchoutput = segnet.predict([curpatch,curweight], batch_size=1, verbose=0)
                else:
                    curpatchoutput = segnet.predict(curpatch, batch_size=1, verbose=0)

                if isinstance(curpatchoutput,list):    # If there are multiple outputs
                    curpatchoutput = curpatchoutput[kerasparas.outID]
                curpatchoutput = np.squeeze(curpatchoutput)

                curpatchoutlabel = curpatchoutput.copy()
                curpatchoutlabel[curpatchoutlabel>=0.5] = 1
                curpatchoutlabel[curpatchoutlabel<0.5] = 0

                curpatchoutlabel = dim_2_categorical(curpatchoutlabel,nclass)
                # lTrans.dim_2_categorical(curpatchoutlabel,nclass)

                categoricalmap[:,i:i+patchdims[0],j:j+patchdims[1],k:k+patchdims[2]] = categoricalmap[:,i:i+patchdims[0],j:j+patchdims[1],k:k+patchdims[2]] + curpatchoutlabel
                likelihoodmap[i:i+patchdims[0],j:j+patchdims[1],k:k+patchdims[2]] = likelihoodmap[i:i+patchdims[0],j:j+patchdims[1],k:k+patchdims[2]] + curpatchoutput
                countermap[i:i+patchdims[0],j:j+patchdims[1],k:k+patchdims[2]] += 1

    for i in range(leng, patchdims[0]-1,-strides[0]):
        for j in range(col, patchdims[1]-1,-strides[1]):
            for k in range(row, patchdims[2]-1,-strides[2]):

                curpatch=img[i-patchdims[0]:i,j-patchdims[1]:j,k-patchdims[2]:k][:].reshape([1,1,patchdims[0],patchdims[1],patchdims[2]])
                for addidx in range(addinputnum):
                    curaddpatch = addinputlist[addidx][i-patchdims[0]:i,j-patchdims[1]:j,k-patchdims[2]:k][:].reshape([1,1,patchdims[0],patchdims[1],patchdims[2]])
                    curpatch = np.append(curpatch,curaddpatch,axis=1)

                if preparas.issubtract:
                    if meanvalue.shape != curpatch.shape:
                        curpatch[:, 0:meanvalue.shape[1], :, :, :] = curpatch[:, 0:meanvalue.shape[1], :, :, :] - meanvalue
                    else:
                        curpatch = curpatch - meanvalue

                if kerasparas.imgformat == 'channels_last':
                    curpatch = np.transpose(curpatch,(0,2,3,4,1))

                if 'W_' in kerasparas.modelname:
                    curweight = np.ones(curpatch.shape,dtype=np.uint8) #shuai
                    curweight = np.ones((curpatch.shape[:4]+(1,)), dtype=np.uint8)  # for breast synseg
                    curpatchoutput = segnet.predict([curpatch,curweight], batch_size=1, verbose=0)
                else:
                    curpatchoutput = segnet.predict(curpatch, batch_size=1, verbose=0)

                if isinstance(curpatchoutput,list):
                    curpatchoutput = curpatchoutput[kerasparas.outID]
                curpatchoutput = np.squeeze(curpatchoutput)

                curpatchoutlabel = curpatchoutput.copy()
                curpatchoutlabel[curpatchoutlabel>=0.5] = 1
                curpatchoutlabel[curpatchoutlabel<0.5] = 0

                curpatchoutlabel = dim_2_categorical(curpatchoutlabel,nclass)
                #curpatchoutlabel = lTrans.dim_2_categorical(curpatchoutlabel,nclass)
                categoricalmap[:,i-patchdims[0]:i,j-patchdims[1]:j,k-patchdims[2]:k] = categoricalmap[:,i-patchdims[0]:i,j-patchdims[1]:j,k-patchdims[2]:k] + curpatchoutlabel
                likelihoodmap[i-patchdims[0]:i,j-patchdims[1]:j,k-patchdims[2]:k] = likelihoodmap[i-patchdims[0]:i,j-patchdims[1]:j,k-patchdims[2]:k] + curpatchoutput
                countermap[i-patchdims[0]:i,j-patchdims[1]:j,k-patchdims[2]:k] +=1
    ####--------------------------------------------------------
    labelmap = np.zeros([leng,col,row],dtype=np.uint8)
    for idx in range(0,leng):
        curslicelabel = np.squeeze(categoricalmap[:, idx,].argmax(axis=0))
        labelmap[idx,] = curslicelabel

    countermap = np.maximum(countermap, 10e-10)
    likelihoodmap = np.divide(likelihoodmap,countermap)

    return labelmap,likelihoodmap

"""Online Evaluation of 3D Image"""
def online_seg_evaluation(imgpath,labelpath,savepath,preparas,normparas,organids,kerasparas):
    # read image file
    imgs = glob.glob(imgpath+"/*.nii.gz")
    print('The number of images processed now is: %d'%len(imgs))

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    organnum = len(organids)
    # build new variables for results
    ppvlists = np.zeros((organnum,len(imgs)))
    senlists = np.zeros((organnum,len(imgs)))
    dsclists = np.zeros((organnum,len(imgs)))
    asdlists = np.zeros((organnum,len(imgs)))

    # load model
    segnet = load_model(kerasparas.modelpath, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

    tab = 0
    for imgname in imgs:
        midname = imgname.rsplit('/',1)[-1]
        if midname.count(".")>0:
            midname=midname[0:midname.index('.')]
        else:
            midname=midname.strip('\n')
        print ('# %d th image name is #: %s' % (tab+1, midname))

        imgraw=sitk.ReadImage(imgpath+"/"+midname+".nii.gz")
        img=sitk.GetArrayFromImage(imgraw)
        normimg=min_max_normalization(img)

        if not os.path.exists(labelpath+"/"+midname+"_mask.nii.gz"):
            print("This image does not have ground truth!!!")
            continue
        labelraw=sitk.ReadImage(labelpath+"/"+midname+"_mask.nii.gz")
        label=sitk.GetArrayFromImage(labelraw)
        label=label_adjustment(label,preparas.nclass,preparas.negids,preparas.sortids)

        if label is None or img.shape!=label.shape:
            print("The shape of the intensity image and ground truth is inconsistent!!!")
            continue

        outlabelmap,outlikelihoodmap=out_LabelHot_map_3D(normimg,segnet,preparas,kerasparas)

        outlabelmapraw = sitk.GetImageFromArray(outlabelmap.astype(np.uint8))
        sitk.WriteImage(outlabelmapraw,os.path.join(savepath + '/%s.nii.gz'%midname))

        # Compute Metrics
        for id in range(organnum):
            dsclists[id,tab] = met.dice(outlabelmap,label,organids[id])
            ppvlists[id,tab] = met.pospreval(outlabelmap,label,organids[id])
            senlists[id,tab] = met.sensitivity(outlabelmap,label,organids[id])
            asdlists[id,tab] = met.asd(outlabelmap,label,organids[id])

        print ('True non-zeros:', np.count_nonzero(label), 'Predict non-zeros:', np.count_nonzero(outlabelmap))
        print ('=======================================')
        for id in range(organnum):
            print ('The dsc of %d organ is: %.4f'%(id+1,dsclists[id,tab]))
            print ('The ppv of %d organ is: %.4f'%(id+1,ppvlists[id,tab]))
            print ('The sen of %d organ is: %.4f'%(id+1,senlists[id,tab]))
            print ('The asd of %d organ is: %.4f'%(id+1,asdlists[id,tab]))
        print ('=======================================')
        tab +=1

    # remove zero value
    dsclists = dsclists[:,:tab]
    ppvlists = ppvlists[:,:tab]
    senlists = senlists[:,:tab]
    asdlists = asdlists[:,:tab]

    print ('Mean dsc of this dataset is:', np.mean(dsclists,axis=1))
    print ('Mean ppv of this dataset is:', np.mean(ppvlists,axis=1))
    print ('Mean sen of this dataset is:', np.mean(senlists,axis=1))
    print ('Mean asd of this dataset is:', np.mean(asdlists,axis=1))

    sio.savemat(savepath+'/%s-%s_%s_metrics.mat'%(preparas.organname,kerasparas.modelname,preparas.stage),{'dsclists': dsclists, 'ppvlists': ppvlists,'senlists':senlists,'asdlists':asdlists})

def online_seg_prediction(imgpath,savepath,preparas,normparas,organids,kerasparas):
    # read image file
    imgs = glob.glob(imgpath+"/*.nii.gz")
    print('The number of images processed now is: %d'%len(imgs))

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    organnum = len(organids)
    # load model
    segnet = load_model(kerasparas.modelpath, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

    tab = 0
    for imgname in imgs:
        tStart = time.time() # time start - liming
        midname = imgname.rsplit('/',1)[-1]
        if midname.count(".")>0:
            midname=midname[0:midname.index('.')]
        else:
            midname=midname.strip('\n')
        print ('# %d the image name is #: %s' % (tab+1, midname))

        imgraw=sitk.ReadImage(imgpath+"/"+midname+".nii.gz")
        img=sitk.GetArrayFromImage(imgraw)
        normimg=min_max_normalization(img)

        outlabelmap,outlikelihoodmap=out_LabelHot_map_3D(normimg,segnet,preparas,kerasparas)

        # Save the results
        outlabelmapraw = sitk.GetImageFromArray(outlabelmap.astype(np.uint8))
        sitk.WriteImage(outlabelmapraw, os.path.join(savepath, f"{midname}_mask.nii.gz"))
        tab+=1
        tEnd = time.time()# time end - liming
        print ('It cost %f sec' % (tEnd - tStart)) #auto round time - liming

def online_seg_prediction_3D(imgpath,savepath,preparas,normparas,organids,kerasparas):
    # read image file
    imgs = glob.glob(imgpath+"/*.nii.gz")
    print('The number of images processed now is: %d'%len(imgs))

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    organnum = len(organids)
    # load model
    segnet = load_model(kerasparas.modelpath, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

    tab = 0
    for imgname in imgs:
        
        midname = imgname.rsplit('/',1)[-1]
        if midname.count(".")>0:
            midname=midname[0:midname.index('.')]
        else:
            midname=midname.strip('\n')
        print ('# %d th image name is #: %s' % (tab+1, midname))
        
        
        imgraw=sitk.ReadImage(imgpath+"/"+midname+".nii.gz")
        img=sitk.GetArrayFromImage(imgraw)
        normimg=min_max_normalization(img)

        outlabelmap,outlikelihoodmap=out_LabelHot_map_3D(normimg,segnet,preparas,kerasparas)

        # Save the results
        outlabelmapraw = sitk.GetImageFromArray(outlabelmap.astype(np.uint8))
        sitk.WriteImage(outlabelmapraw, os.path.join(savepath, f"{midname}_mask.nii.gz"))
        tab+=1
        