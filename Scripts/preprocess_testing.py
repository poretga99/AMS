import numpy as np
import SimpleITK as sitk
import os

task = 'brain-growth'
task_name = 'Task101_BRGR1'

src_path = os.path.join('../Data/test_QUBIQ', task, 'Testing') #'../Data/test_QUBIQ/kidney/Validation'
cases = os.listdir(src_path)

for case in cases:
    tmp = sitk.ReadImage(os.path.join(src_path, case, 'image.nii.gz'))
    tmp_np = sitk.GetArrayFromImage(tmp)
    if len(tmp_np.shape) < 3:
        tmp_np = tmp_np.reshape(1,tmp_np.shape[0], tmp_np.shape[1])
    tmp = sitk.GetImageFromArray(tmp_np)
    sitk.WriteImage(tmp, '/home/mzukovec/Documents/Faks/Semester 3/AMS/nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/'+task_name+'/testing/' + case + '_0000.nii.gz')