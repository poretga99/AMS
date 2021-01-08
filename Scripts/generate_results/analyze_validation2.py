import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

def create_if_nonexistent(path):
    if not os.path.exists(path):
        os.makedirs(path)


data = {
    'organs': ['brain-growth', 'brain-tumor', 'kidney', 'prostate'],
    'numOfTasks': [1, 3, 1, 2],
    'numOfSegsPerTask': [7, 3, 3, 6]
}

internal_task_names = {
    'brain-growth': ['Task101_BRGR1'],
    'brain-tumor': ['Task102_BRTU1', 'Task103_BRTU2', 'Task104_BRTU3'],
    'kidney': ['Task105_KD1'],
    'prostate': ['Task106_PR1', 'Task107_PR2']
}

val_src = '../../Data/test_QUBIQ/'
dest = '../../nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data'

predicted_src = '../../nnUNet/nnunet/nnUNet_Prediction_Results/'
validation_dir = '../../Testing/'
for idx, task in enumerate(internal_task_names):
    for idx2, int_task in enumerate(internal_task_names[task]):
        for case in os.listdir(os.path.join(dest, int_task, 'imagesTs')):
            # Loop through all validation cases
            create_if_nonexistent(os.path.join(validation_dir, task, 'case' + case[:-12][-2:]))
            pred = np.load(os.path.join(predicted_src, int_task, case[:-12]) + '.npz')
            tmp = sitk.ReadImage(os.path.join(val_src, task, 'Testing', 'case' + case[:-12][-2:], 'image.nii.gz'))
            img = (np.array(pred['softmax'][1:, :, :], dtype=np.float32) > 0.5).sum(axis=0) / (
                        data['numOfSegsPerTask'][list(internal_task_names.keys()).index(task)])
            sitk_im = sitk.GetImageFromArray(np.asarray(img, dtype=np.float32))
            sitk.WriteImage(sitk_im, os.path.join(validation_dir, task, 'case' + case[:-12][-2:]) + '/task0'+str(idx2+1)+'.nii.gz')