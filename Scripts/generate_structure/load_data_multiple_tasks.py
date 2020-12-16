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


def parseData(IZZIVI, SRC_PATH, TYPE, NOOFSEGS):
    CASES = [getListOfMasks(appendPath(SRC_PATH, x, TYPE)) for x in izzivi]
    NO_OF_SEGS = {'brain-growth': 1,
                  'brain-tumor': 3,
                  'kidney': 1,
                  'prostate': 2} # Stevilo taskov za dani primer
    d = {}
    for IZZIV in IZZIVI: # create dicts
        d[IZZIV] = {}
    for i in range(len(CASES)):
        IZZIV = IZZIVI[i]
        d[IZZIV][TYPE] = {}
        for CASE in CASES[i]:
            d[IZZIV][TYPE][CASE] = {}
            tmp = os.listdir(SRC_PATH + '/' + IZZIV + '/' + TYPE + '/' + CASE + '/')
            d[IZZIV][TYPE][CASE]['PATHS'] = tmp
            d[IZZIV][TYPE][CASE]['IMAGE'] = [x for x in tmp if 'image' in x]
            d[IZZIV][TYPE][CASE]['TASKS'] = {}
            for j in range(NO_OF_SEGS[IZZIV]):
                check_name = 'task0' + str(j + 1)
                NAME = 'TASK0' + str(j + 1)
                d[IZZIV][TYPE][CASE]['TASKS'][NAME] = [x for x in tmp if check_name in x]
    return d


def saveAverageMasks(data, SOURCE_DIR, TYPE):
    for izziv in data.keys():
        for case in data[izziv][TYPE].keys():
            data[izziv][TYPE][case]['MASKS'] = {}
            for task in data[izziv][TYPE][case]['TASKS'].keys():
                img = sitk.ReadImage('/'.join([SOURCE_DIR, izziv, TYPE, case, 'task01_seg01.nii.gz'])) * 0
                for mask in data[izziv][TYPE][case]['TASKS'][task]:
                    img += sitk.ReadImage('/'.join([SOURCE_DIR, izziv, TYPE, case, mask]))
                img = sitk.Cast(img, sitk.sitkFloat32)
                img /= len(data[izziv][TYPE][case]['TASKS'][task])
                sitk.WriteImage(img, '/'.join([SOURCE_DIR, izziv, TYPE, case, case + '_mask.nii.gz']))
                data[izziv][TYPE][case]['MASKS'][task] = case + '_mask.nii.gz'
    return data


def getPath(*args, **kwargs):
    return '/'.join(args)


def parseIdxToString(idx, length=3):
    tmpArr = []
    tmpArr = ['0' for x in range(length - 1 - int(np.log10(idx)))]
    tmpArr.append(str(idx))
    return ''.join(tmpArr)

def createAverageMasks(srcs, SOURCE, IZZIV, TYPE, CASE):
    oMasks = []
    mask = sitk.GetArrayFromImage(sitk.ReadImage(getPath(SOURCE, IZZIV, TYPE, CASE, srcs[0])))
    for i in range(1, len(srcs)):
        mask += sitk.GetArrayFromImage(sitk.ReadImage(getPath(SOURCE, IZZIV, TYPE, CASE, srcs[i])))
    mask = np.asarray(mask, dtype=np.float32)
    for i in range(len(srcs)):
        tmp = np.asarray(mask >= i + 1, np.float32)
        if len(tmp.shape) < 3:
            tmp.reshape(1, tmp.shape[0], tmp.shape[1])
        oMasks.append(np.asarray(mask >= (i + 1), np.float32))
        # returna maske, kjer je 0-ti element maska z najmanjšo gotovostjo, torej podrocja, kjer se je markiral zgolj en izmed oznacevalcev
    return oMasks

def verifyDimensions(iImage: np.ndarray):
    if len(iImage.shape) < 3:
        iImage.reshape(1,iImage.shape[0], iImage.shape[1])
    return iImage

def makeNNUnetStructure(data, SOURCE, DEST, TYPE):
    mapNames = {'brain-growth': 'BRGR', 'brain-tumor': 'BRTU', 'kidney': 'KD1', 'prostate': 'PR'}
    taskIdx = 200  # Custom datasets starting from 100
    for IZZIV in data.keys():
        if IZZIV == 'kidney':
            noOfTasks = 1
            for TASK in data[IZZIV][TYPE]['case01']['TASKS'].keys():  # Number of different organs
                imageIdx = [1,1,1]
                for CASE in data[IZZIV][TYPE].keys():
                    srcImage = sitk.ReadImage(
                        getPath(SOURCE, IZZIV, TYPE, CASE, 'image.nii.gz'))
                    np_masks = createAverageMasks(data[IZZIV][TYPE][CASE]['TASKS'][TASK], SOURCE, IZZIV, TYPE, CASE)
                    for SEGMENTATION in range(len(data[IZZIV][TYPE][CASE]['TASKS'][TASK])):
                        tmp = sitk.GetArrayFromImage(srcImage)
                        if (len(tmp.shape) > 3) and tmp.shape[0] > 1:
                            print("Error at image ", getPath(SOURCE, IZZIV, TYPE, CASE, 'image.nii.gz'))
                        if len(tmp.shape) < 3:
                            tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1])
                        srcImage = sitk.GetImageFromArray(tmp)
                        sitk.WriteImage(srcImage,
                                        getPath(DEST, 'Task' + str(taskIdx + SEGMENTATION) + '_' + mapNames[IZZIV] + str(noOfTasks + SEGMENTATION),
                                                'imagesTr', mapNames[IZZIV] + str(noOfTasks + SEGMENTATION) + '_' + parseIdxToString(
                                                imageIdx[SEGMENTATION]) + '_0000.nii.gz'))
                        tmp = np_masks[SEGMENTATION]
                        if (len(tmp.shape) < 3):
                            tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1])
                        srcSeg = sitk.GetImageFromArray(tmp)
                        sitk.WriteImage(srcSeg,
                                        getPath(DEST, 'Task' + str(taskIdx + SEGMENTATION) + '_' + mapNames[IZZIV] + str(SEGMENTATION+1),
                                                'labelsTr', mapNames[IZZIV] + str(noOfTasks + SEGMENTATION) + '_' + parseIdxToString(
                                                imageIdx[SEGMENTATION]) + '.nii.gz'))
                        imageIdx[SEGMENTATION] += 1
                noOfTasks += 1
                taskIdx += 1

def generarateJSON(task_name, desc, TASK_SOURCE):
    overwrite_json_file = True  # make it True if you want to overwrite the dataset.json file in Task_folder
    json_file_exist = False
    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = desc
    json_dict['tensorImageSize'] = "2D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"

    # you may mention more than one modality
    json_dict['modality'] = {
        "0": "MRI"
    }
    # labels+1 should be mentioned for all the labels in the dataset

    json_dict['labels'] = {
        "0": "background",
        "1": "mask"
    }

    train_label_dir = TASK_SOURCE + '/imagesTr'
    test_dir = TASK_SOURCE + '/imagesTs'
    train_ids = os.listdir(train_label_dir)
    test_ids = os.listdir(test_dir)
    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)

    # no modality in train image and labels in dataset.json
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i[:i.find('_0000')], "label": "./labelsTr/%s.nii.gz" % i[:i.find('_0000')]} for i in train_ids]

    # removing the modality from test image name to be saved in dataset.json
    json_dict['test'] = ["./imagesTs/%s" % i for i in test_ids]

    with open(os.path.join(TASK_SOURCE, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)


data_paths = {'brain-growth':{},
             'brain-tumor':{},
             'kidney': {},
             'prostate': {}}

izzivi = ['brain-growth', 'brain-tumor', 'kidney', 'prostate'] # izzivi
numOfSegs = [7, 3, 3, 3, 3, 6, 6] # Stevilo segmentacij za posamezen task
desc = ['brain-growth AMS', 'brain-tumor 1 AMS', 'brain-tumor 2 AMS', 'brain-tumor 3 AMS', 'kidney AMS', 'prostate 1 AMS', 'prostate 2 AMS'] # Opis taskov
data = parseData(izzivi, '../../Data/training_data_v2', 'Training', numOfSegs)
print()
#data = saveAverageMasks(data, '../data/training_data_v2', 'Training')
makeNNUnetStructure(data, '../Data/training_data_v2', '../Data/nnUNet/multiple_masks/kidney', 'Training')



#for id, task in enumerate(os.listdir('../Data/nnUNet')):
#    if task == "Task105_KD1":
#        generarateJSON(task, desc[id], '../Data/nnUNet/' + task, task[task.find('_')+1:], 3) # TODO: popravi
generarateJSON('KD11', 'Kidney11', '../Data/nnUNet/multiple_masks/kidney/Task200_KD11')
generarateJSON('KD12', 'Kidney12', '../Data/nnUNet/multiple_masks/kidney/Task201_KD12')
generarateJSON('KD13', 'Kidney13', '../Data/nnUNet/multiple_masks/kidney/Task202_KD13')