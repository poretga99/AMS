import numpy as np
import SimpleITK as sitk
import os

def createIfDontExist(path):
    if not os.path.exists(path):
        os.makedirs(path)

task = 'Task106_PR1'
task_name = 'prostate'

im_src_path = os.path.join('/home/mzukovec/Documents/Faks/Semester 3/AMS/Data/test_QUBIQ', task_name, 'Testing')
tests_path = os.path.join('/home/mzukovec/Documents/Faks/Semester 3/AMS/nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data', task, 'testing')
end_dir = '../Results'



for case in os.listdir(im_src_path):
    if 'case' in case:
        for i in range(2):
            createIfDontExist(os.path.join(end_dir, task_name, case))
            tmp = sitk.ReadImage(os.path.join(im_src_path, case, 'image.nii.gz'))
            tmp_np = sitk.GetArrayFromImage(tmp)
            _, rows, cols = tmp_np.shape
            tmp_np = np.zeros((rows, cols), dtype=np.float32)
            sitk_im = sitk.GetImageFromArray(tmp_np)
            sitk_im.SetOrigin(tmp.GetOrigin())
            sitk_im.SetSpacing(tmp.GetSpacing())
            sitk.WriteImage(sitk_im, os.path.join(end_dir, task_name, case) + '/task0'+str(i+1)+'.nii.gz')