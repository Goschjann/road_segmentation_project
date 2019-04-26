"""
Script to test already trained model on any test dataset

Input: preprocessed test images

Output: .csv with results and visualized images
"""

from __future__ import division, print_function, absolute_import
from keras.models import Model, load_model
from PIL import Image

import numpy as np
import custommodule
import csv

custommodule.checker()

# set path for storage of evaulation images and a summary of the net's scores
test_folder = storage = "/home/jgucci/Desktop/projects/road_segmentation_public/test/preprocessed/"
result_storage_path = "/home/jgucci/Desktop/projects/road_segmentation_public/results/"

# read test data
y_test = custommodule.read_label_masks(path = test_folder,s
                                       mask_name = 'lab',
                                       suffix = '.png',
                                       amount_masks = 165,
                                       crop_pixels = 0,
                                       output_shape = [512, 512],
                                       progress_step = 50,
                                       separation = False)

x_test = custommodule.read_input_maps(path = test_folder,
                                      image_name = 'sat',
                                      suffix = '.png',
                                      amount_maps = 165,
                                      crop_pixels = 0,
                                      output_shape = [512, 512, 3],
                                      progress_step = 50)

# load model
model = load_model('{}mnih_25022019_big.h5'.format(result_storage_path))
model.summary()

# predict and reshape the numpy.ndarrays for the prediction
prediction = model.predict(x_test, batch_size = 2)
prediction_reshaped = prediction.reshape(-1, 512, 512)

# transform the y_test to int64 format
y_test_reshaped = y_test.reshape(-1, 512, 512)
y_test_reshaped = np.ndarray.astype(y_test_reshaped, 'int64')

# transform the satellite data to uint8 rgb array with values in {0, ...,255}
# we solely need this for the visualization via print_four_images_quadratic, thus we call it x_repic
x_repic = np.ndarray.astype(x_test*255, dtype='uint8')


# Hard-Threshold the predicted values with the F1 optimal threshold value
prediction_thresholded = custommodule.threshold_predictions(prediction=prediction_reshaped, threshold=0.4)

# limit for plotting of test images
limit_test = len(x_test)

# initialize lists that store the performance measure values for all predicted images
f1_list = []
acc_list = []
tpr_list = []
prec_list = []
fpr_list = []
npratio_list = []
posratio_list = []

for i in range(limit_test):
    # initialize result class that calculates and stores all evaluation measures
    res = custommodule.result(prediction=prediction_thresholded[i], truevalue=y_test_reshaped[i])
    res.calculate_measures()

    # append to evaluation lists
    f1_list.append(res.f1)
    acc_list.append(res.acc)
    tpr_list.append(res.tpr)
    prec_list.append(res.prec)
    fpr_list.append(res.fpr)
    npratio_list.append((res.neg/res.pos))
    posratio_list.append((res.pos/(res.neg + res.pos)))



    # transfrom to image format for plotting
    x_repic_instance = Image.fromarray(x_repic[i], mode='RGB')

    # visualize prediction depending on comparison of prediction and true labels from results class
    vis = custommodule.visualize_prediction(resultmatrix=res.resultmatrix, x_image=x_repic[i])
    vis_repic = Image.fromarray(vis, mode ='RGB')

    # transfrom to image format for plotting    
    y_repic = np.ndarray.astype(y_test_reshaped[i] * 255, dtype='uint8')
    y_repic = Image.fromarray(y_repic, mode = 'L')

    # the returned probability mask
    pred_repic = np.ndarray.astype(prediction_reshaped[i] * 255, dtype='uint8')
    pred_repic = Image.fromarray(pred_repic, mode = 'L')

    # Plot it quadratic. Other formats can be found in the custommodule
    custommodule.print_four_images_quadratic(header=('test image #' + str(i)), image1=y_repic, image2=pred_repic,image3=vis_repic,
                                   image4= x_repic_instance, title1='true', title2='predicted', title3='coloured satellite',
                                   title4='original satellite', subtitle= str(res.summary), plotit=False,
                                   storagepath=(result_storage_path + 'niceresult' + str(i) + '.png'))


# store the results for the evaluation measures
result_dict ={  'average_f1_score': sum(f1_list) / len(f1_list),
                'average_accuracy': sum(acc_list) / len(acc_list),
                'average_recall': sum(tpr_list) / len(tpr_list),
                'average_precision': sum(prec_list) / len(prec_list),
                'average_fpr': sum(fpr_list) / len(fpr_list),
                'average_negpos_ratio': sum(npratio_list) / len(npratio_list),
                'average_posratio':sum(posratio_list)/len(posratio_list)}

# write to .csv
with open((result_storage_path + 'results.csv'), 'w') as f:
    w = csv.DictWriter(f, result_dict.keys())
    w.writeheader()
    w.writerow(result_dict)

# print the results for the evaluation measures to the command line
print('Average F1 Score: ', sum(f1_list) / len(f1_list))
print('Average Accuracy: ', sum(acc_list) / len(acc_list))
print('Average Recall: ', sum(tpr_list) / len(tpr_list))
print('Average Precision: ', sum(prec_list) / len(prec_list))
print('Average FPR/ 1 - Specificity: ', sum(fpr_list) / len(fpr_list))
print('Average neg/pos-ratio: ', sum(npratio_list) / len(npratio_list))
print('Average posratio', sum(posratio_list) / len(posratio_list))










