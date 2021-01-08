#!/usr/bin/env python
# coding: utf-8

# In[1]:

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

def getAverage(path):
    paths = os.listdir(path)
    tmp = []
    for pt in paths:
        if 'task' in pt:
            tmp.append(pt)
    tmp2 = sitk.ReadImage(os.path.join(path, tmp[0]))
    for i in range(1, len(tmp)):
        tmp2 += sitk.ReadImage(os.path.join(path, tmp[i]))
    return sitk.GetArrayFromImage(tmp2)/7.0


# In[2]:


src = sitk.ReadImage('/home/mzukovec/Documents/Faks/Semester 3/AMS/nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task101_BRGR1/imagesTs/Case35_0000.nii.gz')
seg = sitk.ReadImage('/home/mzukovec/Documents/Faks/Semester 3/AMS/nnUNet/nnunet/nnUNet_Prediction_Results/Task101_BRGR1/Case35.nii.gz')
data = np.load('/home/mzukovec/Documents/Faks/Semester 3/AMS/nnUNet/nnunet/nnUNet_Prediction_Results/Task101_BRGR1/Case35.npz')
ground_truth = getAverage('Data/validation_data_v2/brain-growth/Validation/case35')


# In[3]:


plt.figure()
plt.subplot(1,2,1)
plt.imshow(sitk.GetArrayFromImage(src)[0,:,:], cmap='gray')
plt.title('Reference')
plt.subplot(1,2,2)
plt.title('Predicted')
plt.imshow((np.array(data['softmax'][1:,:,:], dtype=np.float32)>0.5).sum(axis=0), cmap='jet')
plt.show()


# In[4]:


plt.figure()
plt.subplot(1,2,1)
plt.imshow(ground_truth, cmap='jet')
plt.title('Reference')
plt.subplot(1,2,2)
plt.title('Predicted')
plt.imshow((np.array(data['softmax'][1:,:,:], dtype=np.float32)>0.9).sum(axis=0), cmap='jet')
plt.show()


# In[5]:


plt.figure()
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(np.array(data['softmax'][i,:,:]>0.9, dtype=np.float32), cmap='jet')
    plt.title('Strinjanje ' + str(i))
plt.show()


# In[6]:


def dice(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    return dc

def total_dice(ground_truth, predicted):
    prob_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tmp = 0
    for idx, th in enumerate(prob_threshold):
        predicted_thr = (predicted > th).astype(np.uint8)
        ground_truth_thr = (ground_truth > th).astype(np.uint8)
        x = dice(predicted_thr, ground_truth_thr)
        '''
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(ground_truth_thr.squeeze())
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_thr.squeeze())
        plt.title('Dice score {:.2f}'.format(x))
        plt.show()
        '''
        tmp += dice(predicted_thr, ground_truth_thr)
    return tmp/9.0


# In[7]:


ground_truth = getAverage('Data/validation_data_v2/brain-growth/Validation/case35')
predicted= (np.array(data['softmax'][1:,:,:], dtype=np.float32)>1).sum(axis=0)
print(total_dice(ground_truth, predicted))


# In[8]:


ground_truth.max()


# In[9]:


cases = os.listdir('/home/mzukovec/Documents/Faks/Semester 3/AMS/Data/validation_data_v2/brain-growth/Validation')
ground_truths = []
predictions = []

for case in cases:
    ground_truths.append(getAverage('Data/validation_data_v2/brain-growth/Validation/' + case))
    data = np.load('/home/mzukovec/Documents/Faks/Semester 3/AMS/nnUNet/nnunet/nnUNet_Prediction_Results/Task101_BRGR1/'+case+'.npz')
    predictions.append((np.array(data['softmax'][1:,:,:], dtype=np.float32)>0.99).sum(axis=0)/7.0)


# In[10]:


dice_score = []
for i in range(len(ground_truths)):
    dice_score.append(total_dice(ground_truths[i], predictions[i]))

idx = 1
plt.figure()
for i in range(len(ground_truths)):
    plt.subplot(len(ground_truths), 2, idx)
    idx += 1
    plt.imshow(np.array(ground_truths[i], dtype=np.float32), cmap='jet')
    plt.subplot(len(ground_truths), 2, idx)
    idx += 1
    plt.imshow(np.array(predictions[i], dtype=np.float32), cmap='jet')
plt.show()


# In[11]:


print('Dice for validation images: ', np.mean(dice_score))


# In[12]:


dice_score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




