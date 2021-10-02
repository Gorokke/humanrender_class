#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np
import matplotlib
import matplotlib.image as img
from scipy import signal
from scipy import ndimage
import subprocess
import ssim




input_dir = './'
output_dir = './'

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')



if not os.path.isdir(truth_dir):
    print( "%s doesn't exist" % truth_dir)

if not os.path.isdir(submit_dir):
    print( "%s doesn't exist" % submit_dir)


train_list_path=os.path.join(truth_dir, "truth_list.txt")
train_list = np.genfromtxt(train_list_path, dtype=np.str)
result_list_ssim=[]

result_list_rmse=[]


if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


for ii in range(len(train_list)):

    truth_file = os.path.join(truth_dir, train_list[ii]+".jpg")
    truth=img.imread(truth_file).astype(np.float64)

    truth_mask_file = os.path.join(truth_dir, train_list[ii]+"_mask.png")
    truth_mask=img.imread(truth_mask_file).astype(np.float64)
    if np.max(truth_mask)==255:
        truth_mask=truth_mask/255

    truth_mask= np.expand_dims(truth_mask, axis=2)
    truth_mask=np.concatenate((truth_mask, truth_mask,truth_mask), axis=2)




    submission_answer_file = os.path.join(submit_dir,train_list[ii]+ ".jpg")
    predict=img.imread(submission_answer_file).astype(np.float64)


    truth=np.multiply(truth,truth_mask)


    if truth.shape[2]>2:
        truth_gray=(truth[:,:,0]+truth[:,:,1]+truth[:,:,2])/3

    if predict.shape[2]>2:
        predict_gray=(predict[:,:,0]+predict[:,:,1]+predict[:,:,2])/3


    res_ssim = np.mean(ssim.ssim(truth_gray,predict_gray))
    res_rmse=1-np.sqrt(np.mean((truth-predict)**2))/255




    result_list_ssim.append(res_ssim)
    result_list_rmse.append(res_rmse)

# score_file.write("Duration: %0.6f\n" % metadata['elapsedTime'])

avg_rmse=np.mean(result_list_rmse)
avg_ssim=np.mean(result_list_ssim)



print('ssim score: ' , avg_ssim , 'rmse score: ', avg_rmse)
