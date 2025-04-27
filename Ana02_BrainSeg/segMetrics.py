import sys
import numpy as np
from scipy.ndimage import morphology
sys.dont_write_bytecode = True  # don't generate the binray python file .pyc

"""dice coefficient"""
def dice(pre, gt, tid=1):
    pre=pre==tid   #make it boolean
    gt=gt==tid     #make it boolean
    pre=np.asarray(pre).astype(np.bool)
    gt=np.asarray(gt).astype(np.bool)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    dsc=(2. * intersection.sum() + 1e-07) / (pre.sum() + gt.sum() + 1e-07)

    return dsc

"""positive predictive value"""
def pospreval(pre,gt,tid=1):
    pre=pre==tid #make it boolean
    gt=gt==tid   #make it boolean
    pre=np.asarray(pre).astype(np.bool)
    gt=np.asarray(gt).astype(np.bool)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    ppv=(1.0*intersection.sum() + 1e-07) / (pre.sum()+1e-07)

    return ppv

"""sensitivity"""
def sensitivity(pre,gt,tid=1):
    pre=pre==tid #make it boolean
    gt=gt==tid   #make it boolean
    pre=np.asarray(pre).astype(np.bool)
    gt=np.asarray(gt).astype(np.bool)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    sen=(1.0*intersection.sum()+1e-07) / (gt.sum()+1e-07)

    return sen

"""average surface distance"""
def surfd(pre, gt, tid=1, sampling=1, connectivity=1):
    pre=pre==tid   #make it boolean
    gt=gt==tid     #make it boolean

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    input_1 = np.atleast_1d(pre.astype(np.bool))
    input_2 = np.atleast_1d(gt.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = np.logical_xor(input_1,morphology.binary_erosion(input_1, conn))
    Sprime = np.logical_xor(input_2,morphology.binary_erosion(input_2, conn))

    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)

    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
    return sds

def asd(pre, gt, tid=1, sampling=1, connectivity=1):
    sds = surfd(pre, gt, tid=tid, sampling=sampling, connectivity=connectivity)
    dis = sds.mean()
    return dis