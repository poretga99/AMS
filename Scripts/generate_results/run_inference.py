import os
import sys

os.chdir('../../nnUNet/nnunet')
for task in os.listdir(os.path.join('nnUNet_raw_data_base/nnUNet_raw_data')):
    os.system('nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/{}/imagesTs -o nnUNet_Prediction_Results/{} -t {} -tr nnUNetTrainerV2_Loss_Dice_Soft -m 2d --save_npz --overwrite_existing --f all'.format(task, task, int(task[4:7])))