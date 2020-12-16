import numpy as np
import SimpleITK as sitk
import os


src_path = '../Data/validation_data_v2/kidney/Validation'
folders = os.listdir(src_path)

for folder in folders:
    ims = os.listdir(os.path.join(src_path, folder))
    for im in ims:
        if im == 'image.nii.gz':
            tmp = sitk.ReadImage(os.path.join(src_path, folder, 'image.nii.gz'))
            tmp_np = sitk.GetArrayFromImage(tmp)
            if len(tmp_np.shape) < 3:
                tmp_np = tmp_np.reshape(1,tmp_np.shape[0], tmp_np.shape[1])
            tmp = sitk.GetImageFromArray(tmp_np)
            sitk.WriteImage(tmp, '/home/mzukovec/Documents/Faks/Semester 3/AMS/nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task105_KD1/imagesTs/' + folder + '_0000.nii.gz')