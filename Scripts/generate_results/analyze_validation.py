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

val_src = '../../Data/validation_data_v2/'
dest = '../../nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data'

for idx, organ in enumerate(data['organs']):
    for task in range(data['numOfTasks'][idx]):
        # Prepare testing images and put them in the right directory
        src = os.path.join(val_src, organ, 'Validation')
        int_task = internal_task_names[organ][task]
        for case in os.listdir(src):
            # Loop through all cases in validation dataset
            tmp = sitk.ReadImage(os.path.join(src, case, 'image.nii.gz'))
            tmp_np = sitk.GetArrayFromImage(tmp)
            if len(tmp_np.shape) < 3:
                tmp_np = tmp_np.reshape(1, tmp_np.shape[0], tmp_np.shape[1])
            tmp = sitk.GetImageFromArray(tmp_np)
            if organ == 'brain-tumor':
                for i in range(4):
                    tmp = sitk.GetImageFromArray(tmp_np[i,:,:].reshape(1,tmp_np.shape[1],tmp_np.shape[2]))
                    sitk.WriteImage(tmp, os.path.join(dest, int_task, 'imagesTs', case + '_000'+str(i)+'.nii.gz'))
            sitk.WriteImage(tmp, os.path.join(dest, int_task, 'imagesTs', case + '_0000.nii.gz'))
        # Run inference on testing images
        os.system(
            'nnUNet_predict -i  ../../nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/{task}/imagesTs -o  ../../nnUNet/nnunet/nnUNet_Prediction_Results/{task} -t {task_No} -tr nnUNetTrainerV2_Loss_Dice_Soft -m 2d --save_npz --overwrite_existing'.format(
                task=int_task, task_No=int_task[4:7]))

predicted_src = '../../nnUNet/nnunet/nnUNet_Prediction_Results/'
validation_dir = '../../Validation/'
for idx, task in enumerate(internal_task_names):
    for idx2, int_task in enumerate(internal_task_names[task]):
        for case in os.listdir(os.path.join(dest, int_task, 'imagesTs')):
            # Loop through all validation cases
            create_if_nonexistent(os.path.join(validation_dir, task, case[:-12]))

            pred = np.load(os.path.join(predicted_src, int_task, case[:-12]) + '.npz')
            tmp = sitk.ReadImage(os.path.join(val_src, task, 'Validation', case[:-12], 'task01_seg01.nii.gz'))
            img = (np.array(pred['softmax'][1:, :, :], dtype=np.float32) > 0.5).sum(axis=0) / (
                        data['numOfSegsPerTask'][list(internal_task_names.keys()).index(task)])
            if tmp.GetDepth():
                img = img.reshape(tmp.GetDepth(), tmp.GetHeight(), tmp.GetWidth())
            sitk_im = sitk.GetImageFromArray(np.asarray(img, dtype=np.float32))
            sitk_im.SetOrigin(tmp.GetOrigin())
            sitk_im.SetSpacing(tmp.GetSpacing())
            sitk_im.SetDirection(tmp.GetDirection())
            sitk.WriteImage(sitk_im, os.path.join(validation_dir, task, case[:-12]) + '/task0'+str(idx2+1)+'.nii.gz')