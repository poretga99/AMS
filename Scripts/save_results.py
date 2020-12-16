import numpy as np
import SimpleITK as sitk
import os

def createIfDontExist(path):
    if not os.path.exists(path):
        os.makedirs(path)

task = 'Task105_KD1'
task_name = 'kidney'

im_src_path = os.path.join('/home/mzukovec/Documents/Faks/Semester 3/AMS/Data/test_QUBIQ', task_name, 'Testing')
src_path = os.path.join('/home/mzukovec/Documents/Faks/Semester 3/AMS/nnUNet/nnunet/nnUNet_Prediction_Results', task)
tests_path = os.path.join('/home/mzukovec/Documents/Faks/Semester 3/AMS/nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data', task, 'testing')
end_dir = '../Results'



for case in os.listdir(tests_path):
    if 'case' in case:
        createIfDontExist(os.path.join(end_dir, task_name, case[:-12]))
        data = np.load(os.path.join(src_path, case[:-12]) + '.npz')
        img = (np.array(data['softmax'][1:, :, :], dtype=np.float32) > 0.5).sum(axis=0)/(data['softmax'].shape[0]-1)
        sitk_im = sitk.GetImageFromArray(np.asarray(img, dtype=np.float32))
        tmp = sitk.ReadImage(os.path.join(im_src_path, case[:-12], 'image.nii.gz'))
        sitk_im.SetOrigin(tmp.GetOrigin())
        sitk_im.SetSpacing(tmp.GetSpacing())
        sitk_im.SetDirection(tmp.GetDirection())
        sitk.WriteImage(sitk_im, os.path.join(end_dir, task_name, case[:-12]) + '/task01.nii.gz')