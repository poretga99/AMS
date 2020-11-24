import os
import numpy as np
import SimpleITK as itk
from tqdm import tqdm
from keras import backend as K

from os.path import join

def resample_image(input_image, spacing_mm=(1, 1), spacing_image=None, inter_type=itk.sitkLinear):
    """
    Resample image to desired pixel spacing.

    Should specify destination spacing immediate value in parameter spacing_mm or as SimpleITK.Image in 
    spacing_image. You must specify either spacing_mm or spacing_image, not both at the same time.

    :param input_image: Image to resample.
    :param spacing_mm: Spacing for resampling in mm given as tuple or list of two/three (2D/3D) float values.
    :param spacing_image: Spacing for resampling taken from the given SimpleITK.Image.
    :param inter_type: Interpolation type using one of the following options:
                            SimpleITK.sitkNearestNeighbor,
                            SimpleITK.sitkLinear,
                            SimpleITK.sitkBSpline,
                            SimpleITK.sitkGaussian,
                            SimpleITK.sitkLabelGaussian,
                            SimpleITK.sitkHammingWindowedSinc,
                            SimpleITK.sitkBlackmanWindowedSinc,
                            SimpleITK.sitkCosineWindowedSinc,
                            SimpleITK.sitkWelchWindowedSinc,
                            SimpleITK.sitkLanczosWindowedSinc
    :type input_image: SimpleITK.Image
    :type spacing_mm: Tuple[float]
    :type spacing_image: SimpleITK.Image
    :type inter_type: int
    :rtype: SimpleITK.Image
    :return: Resampled image as SimpleITK.Image.
    """
    resampler = itk.ResampleImageFilter()
    resampler.SetInterpolator(inter_type)

    if (spacing_mm is None and spacing_image is None) or \
       (spacing_mm is not None and spacing_image is not None):
        raise ValueError('You must specify either spacing_mm or spacing_image, not both at the same time.')

    if spacing_image is not None:
        spacing_mm = spacing_image.GetSpacing()

    input_spacing = input_image.GetSpacing()
    # set desired spacing
    resampler.SetOutputSpacing(spacing_mm)
    # compute and set output size
    output_size = np.array(input_image.GetSize()) * np.array(input_spacing) \
                  / np.array(spacing_mm)
    output_size = list((output_size + 0.5).astype('uint32'))
    output_size = [int(size) for size in output_size]
    resampler.SetSize(output_size)

    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())

    resampled_image = resampler.Execute(input_image)

    return resampled_image


def load_kidney_data(output_size=(256, 256), new_spacing_mm=None, region_of_interest_size=(256, 256), 
                     region_of_interest_index=(50, 200), dir_path='./data/train/kidney'):
    """
    Load data from dir_path folder in the format suitable for training neural networks. 
    This function is an EXAMPLE and should NOT be considered best practice.

    This function will load all images from a given directory `dir_path`, perform cropping to fixed 
    size and resample the obtained image such that the output size will match the one specified 
    by parameter 'output_size'. For each image a random mask will be chosen and cropped to the same
    size as image.
    
    :param output_size: Define output image size.
    :param new_spacing_mm: Define output image spacing. Ignored if `output_size` is given.
    :param region_of_interest_size: Size in pixels of the region extracted.
    :param region_of_interest_index: The inclusive starting index of the region extracted.
    :param dir_path: Define output image spacing. Ignored if `output_size` is given.

    :type output_size: tuple[int]
    :type new_spacing_mm: tuple[float]
    :type region_of_interest_size: tuple[int]
    :type region_of_interest_index: tuple[int]
    :type dir_path: str

    
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :return: Image data and kidney mask in a tuple.
    """
    # hidden function parameters
    DATA_PATH = dir_path
    

    def extract_image(image, output_size=None, new_spacing_mm=None, interpolation_type=itk.sitkLinear,
                      region_of_interest_size=None, region_of_interest_index=None):
        """
        Define image extraction function based on cropping and resampling, with either the size or spacing of the
        output fixed. If parameter `output_size` is given then `new_spacing_mm` is ignored.
        """
        # crop image for a given region of interest
        if region_of_interest_size is not None and region_of_interest_index is not None:
            image = itk.RegionOfInterest(image, region_of_interest_size, region_of_interest_index)
        
        if output_size is not None:
            imsize = image.GetSize()
            new_spacing_mm = (imsize[0] / output_size[0], imsize[1] / output_size[1])

        return resample_image(
            image, 
            spacing_mm = new_spacing_mm, 
            inter_type=interpolation_type)
    
    
    # check for required parameters
    if output_size is None and new_spacing is None:
        raise ValueError('Parameter `output_size` or `new_spacing_mm` must be given.')

    # load and extract all images and masks into a list of dicts
    kidney_data = []
    patient_paths = sorted([f for f in os.listdir(DATA_PATH) if not f.startswith('.')], key=str.lower)
    
    for pacient_no in tqdm(range(len(patient_paths))):
        patient_path = join(DATA_PATH, patient_paths[pacient_no])

        # read image and a random mask
        image = itk.ReadImage(join(patient_path, 'image.nii.gz'))
        seg1 = itk.ReadImage(join(patient_path, 'task01_seg01.nii.gz'))
        seg2 = itk.ReadImage(join(patient_path, 'task01_seg02.nii.gz'))
        seg3 = itk.ReadImage(join(patient_path, 'task01_seg03.nii.gz'))

        # crop and resample the images 
        image = extract_image(image, output_size=output_size, new_spacing_mm=new_spacing_mm, 
                             interpolation_type=itk.sitkLinear, region_of_interest_size=region_of_interest_size,
                             region_of_interest_index=region_of_interest_index) 
        seg1 = extract_image(seg1, output_size=output_size, new_spacing_mm=new_spacing_mm, 
                             interpolation_type= itk.sitkNearestNeighbor, region_of_interest_size=region_of_interest_size,
                             region_of_interest_index=region_of_interest_index)
        seg2 = extract_image(seg2, output_size=output_size, new_spacing_mm=new_spacing_mm, 
                             interpolation_type= itk.sitkNearestNeighbor, region_of_interest_size=region_of_interest_size,
                             region_of_interest_index=region_of_interest_index)
        seg3 = extract_image(seg3,output_size=output_size, new_spacing_mm=new_spacing_mm, 
                             interpolation_type= itk.sitkNearestNeighbor, region_of_interest_size=region_of_interest_size,
                             region_of_interest_index=region_of_interest_index)

        # add to dict 
        kidney_data.append({'image':image, 'seg1':seg1, 'seg2':seg2 ,'seg3':seg3})
        
    # reshape all modalities and masks into 3d arrays
    image_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['image'])) for data in kidney_data])
    seg1_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['seg1'])) for data in kidney_data])
    seg2_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['seg2'])) for data in kidney_data])
    seg3_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['seg3'])) for data in kidney_data])
    
    # combine
    image_array = np.concatenate((image_array, image_array, image_array), axis=2)
    seg_array = np.concatenate((seg1_array, seg2_array, seg3_array), axis=2)
    
    # reshape the 3d arrays such that the number of cases is in the first column 
    image_array = np.transpose(image_array, (2, 0, 1))
    seg_array = np.transpose(seg_array, (2, 0, 1))
     
    # reshape the 3d arrays according to the Keras backend
    if K.image_data_format() == 'channels_first': 
        # this format is (n_cases, n_channels, image_height, image_width)
        image_karray = image_array[:, np.newaxis, :, :]
        seg_karray = seg_array[:, np.newaxis, :, :]
        channel_axis = 1
    else:
        # this format is (n_cases, image_height, image_width, n_channels)
        image_karray = image_array[:, :, :, np.newaxis]
        seg_karray = seg_array[:, :, :, np.newaxis]

        channel_axis = -1

    # get mean and std per image
    m = np.mean(image_karray, axis=(1,2), keepdims=True)
    std = np.std(image_karray, axis=(1,2), keepdims=True)

    # standardize image intensities ~ N(0, 1)
    X = (image_karray - m) / (std + 1e-07)

    # return the image and masks
    return X, seg_karray