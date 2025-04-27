"""Defined Class for Transferring Parameters"""
class KerasParas:
    def __init__(self):
        self.modelname = ''
        self.modelpath = None
        self.weightpath = None
        self.outID = 0                          # indicating which output we need when dealing with multi-output network
        self.thd = 0.5
        self.imgformat = 'channels_first'
        self.loss = None

class PreParas:
    def __init__(self):
        self.patchdims = []
        self.patchlabeldims = []
        self.patchstrides = []
        self.nclass = ''
        self.organname = ''
        self.stage = ''
        self.meanvalue = None
        self.negids = []
        self.sortids = []
        self.saveflag = ''
        self.issubtract = 1

class NormParas:
    def __init__(self):
        self.method = ''
        self.quantile = None
        self.lmin = None
        self.rmax = None
        self.dividend = None
        self.meanvalue = None
