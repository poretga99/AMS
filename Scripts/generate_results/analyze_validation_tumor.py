import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

def create_if_nonexistent(path):
    if not os.path.exists(path):
        os.makedirs(path)

tasks = [['Task201_BRTU21', 'Task202_BRTU22', 'Task203_BRTU23'], ['Task204_BRTU31', 'Task205_BRTU32', 'Task206_BRTU33']]

val_src = '../../Data/validation_data_v2/'
dest = '../../nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data'


for listOfTasks in tasks:
    for task in listOfTasks:
        for i in range(3):
            src = os.path.join(val_src, 'brain-growth', 'Validation')
            for case in os.listdir(src):
                tmp = sitk.ReadImage(os.path.join(src, case, 'image.nii.gz'))
                tmp_np = sitk.GetArrayFromImage(tmp)
                for i in range(4):
                    tmp = sitk.GetImageFromArray(tmp_np[i, :, :].reshape(1, tmp_np.shape[1], tmp_np.shape[2]))
                    sitk.WriteImage(tmp, os.path.join(dest, task, 'imagesTs', case + '_000' + str(i) + '.nii.gz'))
            os.system(
                'nnUNet_predict -i  ../../nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/{task}/imagesTs -o  ../../nnUNet/nnunet/nnUNet_Prediction_Results/{task} -t {task_No} -tr nnUNetTrainerV2_Loss_Dice_Soft -m 2d --save_npz --overwrite_existing'.format(
                    task=task, task_No=task[4:7]))


predicted_src = '../../nnUNet/nnunet/nnUNet_Prediction_Results/'
validation_dir = '../../Validation/'

for sidx, subtask in enumerate(tasks):
    for case in os.path.join(val_src, 'brain-growth', 'Validation'):
        create_if_nonexistent(os.path.join(validation_dir, 'brain-growth_' + str(sidx + 1), case[:-12]))
        predictions = []
        for i in range(3):
            pred = np.load(os.path.join(predicted_src, subtask[i], case[:-12]) + '.npz')
            predictions.append((np.array(pred['softmax'][1:, :, :], dtype=np.float32) > 0.5).sum(axis=0))
        

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