"""
Script to test already trained model on any dataset

Input: test images

Outputs .csv for given list iof thresholds


"""



from __future__ import division, print_function, absolute_import
from PIL import Image
from keras.models import Model, load_model

import numpy as np
import custommodule
import csv

from PIL import Image

custommodule.checker()

# set path for storage of a summary of the nets scores for different thresholds
result_storage_path = '../data/test_results/'

## read test data
y_test = custommodule.read_label_masks(path = '../data/test_data/',
                                       mask_name = 'lab',
                                       suffix = '.png',
                                       amount_masks = 40,
                                       crop_pixels = 0,
                                       output_shape = [512, 512],
                                       progress_step = 10,
                                       separation = False)

x_test = custommodule.read_input_maps(path = '../data/test_data/,
                                      image_name = 'sat',
                                      suffix = '.png',
                                      amount_maps = 40,
                                      crop_pixels = 0,
                                      output_shape = [512, 512, 3],
                                      progress_step = 10)


# load model
model = load_model('maps_005_split_800.h5')

# predict and reshape the numpy.ndarrays for the prediction
prediction = model.predict(x_test, batch_size = 2)
prediction_reshaped = prediction.reshape(-1, 512, 512)

# transform the y_test to int64 format
y_test_reshaped = y_test.reshape(-1, 512, 512)
y_test_reshaped = np.ndarray.astype(y_test_reshaped, 'int64')

# transform the satellite data to uint8 rgb array with values in {0, ...,255}
# we solely need this for the visualization via visualize_prediction(), thus we call it x_repic
x_repic = np.ndarray.astype(x_test*255, dtype='uint8')

# input thresholds that should be evaluated
threshold_list = [0.2, 0.25 , 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

# initialize list for results
result_dict_list = []

for threshold in threshold_list:

    # update user
    print('calculating measures for threshold ', str(threshold))

    # Hard-Threshold the predicted values
    prediction_thresholded = custommodule.threshold_predictions(prediction=prediction_reshaped, threshold=threshold)

    # limit for amount of test images
    # TODO: automate this information retrieval
    limit_test = 40

    # initialize lists that store the values for the evaluation measures for each predicted image
    f1_list = []
    acc_list = []
    tpr_list = []
    prec_list = []
    fpr_list = []
    npratio_list = []

    for i in range(limit_test):
        res = custommodule.result(prediction=prediction_thresholded[i], truevalue=y_test_reshaped[i])
        res.calculate_measures()

        # append to evaluation lists
        f1_list.append(res.f1)
        acc_list.append(res.acc)
        tpr_list.append(res.tpr)
        prec_list.append(res.prec)
        fpr_list.append(res.fpr)
        npratio_list.append((res.neg/res.pos))

    # store the results
    result_dict ={  'threshold': threshold,
                    'average_f1_score': sum(f1_list) / len(f1_list),
                    'average_accuracy': sum(acc_list) / len(acc_list),
                    'average_recall': sum(tpr_list) / len(tpr_list),
                    'average_precision': sum(prec_list) / len(prec_list),
                    'average_fpr': sum(fpr_list) / len(fpr_list),
                    'average_negpos_ratio': sum(npratio_list) / len(npratio_list)}
    # append current result to list
    result_dict_list.append(result_dict)

# write results to .csv file
with open((result_storage_path + 'results_thresholds.csv'), 'w') as f:
    w = csv.DictWriter(f, result_dict_list[0].keys())
    w.writeheader()
    for element in result_dict_list:
        w.writerow(element)












