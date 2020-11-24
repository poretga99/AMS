import os

main_dir = 'C:/Users/zukov/Documents/Faks/Semester 3/AMS/Izziv/Programiranje/nnUNet/nnunet'

os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir,'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed'] = os.path.join(main_dir,'preprocessed')
os.environ['RESULTS_FOLDER'] = os.path.join(main_dir,'nnUNet_trained_models')

