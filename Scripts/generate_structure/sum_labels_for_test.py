import sys
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil
from collections import OrderedDict
import json


def getListOfMasks(SOURCE):
    return os.listdir(SOURCE)


def appendPath(PATH, ORGAN, TYPE):
    return PATH + "/" + ORGAN + "/" + TYPE

def parseIdxToString(idx, length=3):
    tmpArr = []
    tmpArr = ['0' for x in range(length - 1 - int(np.log10(idx)))]
    tmpArr.append(str(idx))
    return ''.join(tmpArr)

izzivi = ['brain-growth', 'kidney', 'prostate'] # izzivi
mapIzzivi = [['Task101_BRGR1'], ['Task105_KD1'], ['Task106_PR1', 'Task107_PR2']]
noOfTasks = [1, 1, 2]
numOfSegs = [7, 3, 3, 3, 3, 6, 6] # Stevilo segmentacij za posamezen task
src = '../../Data/test_QUBIQ/'
dst = '../../nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/'

for izziv in range(len(izzivi)):
    for task in range(noOfTasks[izziv]):
        mapName = mapIzzivi[izziv][task]
        for case in os.listdir(os.path.join(src, izzivi[izziv], 'Testing')):
            src_im = sitk.ReadImage(os.path.join(src, izzivi[izziv], 'Testing', case, 'image.nii.gz'))
            tmp = sitk.GetArrayFromImage(src_im)
            if len(tmp.shape) < 3:
                tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1])
            src_im = sitk.GetImageFromArray(tmp)
            sitk.WriteImage(src_im, os.path.join(dst, mapName, 'imagesTs', mapName[8:]+'_' + str(parseIdxToString(int(case[-2:]))) + '_0000.nii.gz'))

mapIzzivi = ['Task102_BRTU1', 'Task103_BRTU2', 'Task104_BRTU3']
for task in range(3):
    mapName = mapIzzivi[task]
    for case in os.listdir(os.path.join(src, 'brain-tumor', 'Testing')):
        src_im = sitk.ReadImage(os.path.join(src, 'brain-tumor', 'Testing', case, 'image.nii.gz'))
        tmp = sitk.GetArrayFromImage(src_im)
        for i in range(4):
            tmp2 = tmp[i, :, :].reshape(1, tmp.shape[1], tmp.shape[2])
            src_im = sitk.GetImageFromArray(tmp2)
            sitk.WriteImage(src_im, os.path.join(dst, mapName, 'imagesTs', mapName[8:]+'_' + str(parseIdxToString(int(case[-2:]))) + '_000'+str(i)+'.nii.gz'))