"""
This is the local module for the road segmentation
with the purpose to encapsulate all self-written functions and classes in a 
separate local module in order to make our Coding more readable and elegant. 

Summer 2017
"""

import os
import skimage
import PIL
import numpy as np
import skimage.io
import collections
import random

from skimage.transform import rescale
from sklearn import model_selection
from scipy import ndimage
from PIL import Image
from PIL import ImageDraw


#---------------------------------
def checker():
    """
    checks if the library is loaded successfully
    :return: printed message
    """
    print('module succesfully loaded')

#------------------------------------


######################################################################
#
#   PREPROCESSING
#
######################################################################


#---------------------------------

def read_split_save_many(satellite, label, images_path ,store_path, size_crop = 64, thresholdgreen = 0,
                         threshold = .3, amount_inputs = 1, progress = True, augment = False, usetiffs = False):
    """
    Function that splits amount_inputs images and labels, slides them into patches
    and stores those patches that do contain more than threshold % of positive labels
    also, it is able to augment the data with the factor 6 using rotation and flipping of
    the selected images - this is controled by the boolean augment = False
    also, the DEFORESTES is implemented: a thresholdgreen that only selects those images
    that have an mean in the 2nd layer of the matrix, the green layer, above thresholdgreen

    Example: for 1 input set
     size_crop = 640, augment = True -> 8 augmented and not cropped images
     size_crop = 128 (=640/5), augment = False -> 25 cropped but not augmented patches

    Parameters:
    ---------
    :param satellite: the prefix of each satellite image in the images_path, depends on getMapsFunction.
    :param label: the prefix of each label image in the images_path, depends on getMapsFunction.
    :param images_path: path of the raw images as returned by getMapsFunction.
    :param store_path: desired path for the preprocessed data.
    :param size_crop: size of the cropped patches. Set to 640 if no cropping is desired. Should be a valid divisor of
    the original dimensions.
    :param thresholdgreen: percentage of pixels on the satellite image that have to differ from forest pattern.
    Still experimental
    :param threshold: minimum percentage of road pixels for each image to be passed. Option to sort out non-road input data.
    :param amount_inputs: amount of image sets that are passed to the function. Will be automated soon.
    :param progress: boolean that allows to inform the user on the progress of the function.
    :param augment: boolean for dataset augmentation. If set to TRUE, amound of data is increased by factor 8 via image
    flipping and mirroring
    :return: stores preprocessed images in a new folder.
    """


    # loop over amount_inputs images in the input folder
    progress_counter = 0
    # counter for storage of the images
    # counter-intuitively (haha) we start with 0 (Python standard)
    counter = 0

    for l in range(amount_inputs):
        # inform user about progress
        if progress:
            if l/amount_inputs > progress_counter:
                print(str(100*l/amount_inputs) + '% of images processed')
                progress_counter = progress_counter + 0.1
        # read images
        if usetiffs:
            sat = PIL.Image.open((images_path + satellite + str(l) + '.tiff'))
            lab = PIL.Image.open((images_path + label + str(l) + '.tiff'))
        else:
            sat = PIL.Image.open((images_path + satellite + str(l) + '.png'))
            lab = PIL.Image.open((images_path + label + str(l) + '.png'))
        height = size_crop
        width = size_crop
        imgwidth, imgheight = sat.size

        # double loop to run over grid
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
                # box for the current cut
                box = (j, i, j+width, i+height)
                crop_sat = sat.crop(box)
                crop_lab = lab.crop(box)
                greenlayersat = np.array(crop_sat)[:,:,1]
                # only store those images with more than threshold% positive pixels
                crop_lab_array = np.array(crop_lab)
                thresholder = lambda x: 0 if x == 0 else 1
                # vectorization makes computation faster
                vfunc = np.vectorize(thresholder)
                crop_lab_array_thresh = vfunc(crop_lab_array)
                # threshold for road pixels and forestness
                if crop_lab_array_thresh.mean() >= threshold and greenlayersat.mean() >= thresholdgreen:
                    # option for dataset augmentation
                    if augment:
                        # transform to array
                        crop_sat_array = np.array(crop_sat)

                        # loop for rotation, re-transformation in image and storage
                        for r in range(0,4):

                            ## Augmentation
                            # original, without mirroring, just rotation
                            rot_sat = np.rot90(crop_sat_array, k = r)
                            rot_lab = np.rot90(crop_lab_array_thresh, k = r)

                            # mirrored, rotated images
                            # flipped vertically here (axis = 0, axis = 1 for horizontal)
                            flip_sat = np.flip(rot_sat, axis=0)
                            flip_lab = np.flip(rot_lab, axis=0)

                            ## Storage
                            # non mirrored
                            Image.fromarray(np.uint8(rot_sat)).save((store_path + 'sat' + str(counter) + '.png'),'PNG')
                            # factor 255 for thresholded label
                            Image.fromarray(np.uint8(rot_lab*255)).save((store_path + 'lab' + str(counter) + '.png'),'PNG')
                            counter += 1

                            # vertically flipped (mirrored)
                            Image.fromarray(np.uint8(flip_sat)).save((store_path + 'sat' + str(counter) + '.png'), 'PNG')
                            Image.fromarray(np.uint8(flip_lab * 255)).save((store_path + 'lab' + str(counter) + '.png'), 'PNG')
                            counter +=1
                    else:
                        crop_sat.save((store_path + 'sat' + str(counter) + '.png'),'PNG')
                        # important to resize as the array is normalized to [0,1]
                        crop_lab = Image.fromarray(np.uint8(crop_lab_array_thresh * 255))
                        crop_lab.save((store_path + 'lab' + str(counter) + '.png'),'PNG')
                        counter += 1


# ---------------------------------

def split_train_test(path_all, path_splits, ratio_train, dimension = 640, seed = 123, progress = True):

    """
    Splits preprocessed data in test and train data sample. Also, names the reuslting data with coherent numbering.
    Input is folder with all preprocessed data and the output are two subfolders train_data, test_data.

    :param path_all: path of all input images.
    :param path_splits: parent folder for the resulting data folders (usually same as path_all).
    :param ratio_train: ratio of train data relative to test data.
    :param dimension: dimensionality of the images.
    :param seed: random seed for splitting.
    :param progress: inform user about progress or not.
    :return: creates train and test data folder in desired path.
    """

    print('start with splitting')
    # get amount of files in path_all
    amount_files = int(len(os.listdir(path_all)) *0.5)

    all_lab_list = []
    all_sat_list = []

    progress_counter = 0
    for i in range(amount_files):

        # inform user about progress
        if progress:
            if i / amount_files > progress_counter:
                print(str(100 * i / amount_files) + '% of images read in')
                progress_counter = progress_counter + 0.1

        # read in labels and satellites
        instance_lab = skimage.io.imread(fname=(path_all + 'lab' + str(i) + '.png'), as_grey=True)
        instance_sat = skimage.io.imread(fname=(path_all + 'sat' + str(i) + '.png'), as_grey=False)

        # append them to the lists
        all_lab_list.append(instance_lab)
        all_sat_list.append(instance_sat)

    # necessary list formating
    all_lab_list = np.stack(all_lab_list)
    all_sat_list = np.stack(all_sat_list)

    # satellites are 3-dimensional, labels not
    all_lab_list = all_lab_list.reshape(-1, dimension, dimension)
    all_sat_list = all_sat_list.reshape(-1, dimension, dimension, 3)

    # use sklearns inbuild function
    train_sat_list, test_sat_list, train_lab_list, test_lab_list = model_selection.train_test_split(all_sat_list, all_lab_list, train_size=ratio_train, random_state=seed)

    ## store train data
    # create folder
    train_path = (path_splits + 'train_data/')
    os.makedirs(train_path)

    progress_counter = 0
    for j in range(int(amount_files*ratio_train)):
        # transfrom label values from 0-1 to 0-255
        lab = Image.fromarray(np.uint8(train_lab_list[j,:,:]))
        sat = Image.fromarray(train_sat_list[j,:,:,:])
        lab.save((train_path + 'train_lab' + str(j) + '.png'))
        sat.save((train_path + 'train_sat' + str(j) + '.png'))

        # inform user about progress
        if progress:
            if j / (amount_files*ratio_train) > progress_counter:
                print( str(round(100 * j / (amount_files * (ratio_train)),0)) + '% of train data stored')
                progress_counter = progress_counter + 0.1

    ## store test data
    # create folder
    test_path = (path_splits + 'test_data/')
    os.makedirs(test_path)

    progress_counter = 0
    for j in range(int(amount_files * (1-ratio_train))):
        # transfrom label values from 0-1 to 0-255
        lab = Image.fromarray(np.uint8(test_lab_list[j, :, :]))
        sat = Image.fromarray(test_sat_list[j, :, :, :])
        lab.save((test_path + 'test_lab' + str(j) + '.png'))
        sat.save((test_path + 'test_sat' + str(j) + '.png'))

        # inform user about progress
        if progress:
            if j / (amount_files * (1-ratio_train)) > progress_counter:
                print( str(round(100 * j / (amount_files * (1-ratio_train)), 0)) + '% of test data stored')
                progress_counter = progress_counter + 0.1

# --------------------------------

######################################################################
#
#   TRAINING
#
######################################################################


def gaussit(input_list, sigma=(2, 2, 0)):
    """
    Function that takes list of numpy arrays (each element = one satellite image, e.g. output from read_input_maps) and
    returns list of numpy arrays that contain a gaussian version of the images. Noise can be controlled via the sigma()
    argument. in RGB images 3 inputs (2,2,0) for each layer are allowed. Also one integer is ok.

    :param input_list: list of numpy arrays
    :param sigma: value for the gaussian filtering of the the data. Standard deviation of the bell curve.
    :return: list of numpy arrays that were filtered with the gauss filter.
    """
    output_list = []
    for image in input_list:
        output_list.append(ndimage.gaussian_filter(input=image, sigma=sigma))
    output_list = np.asarray(output_list)
    output_list = np.ndarray.astype(output_list, 'float32')
    return output_list

# ---------------------------------


# Generic Function that reads Input Map data from given folder path and amount of maps, crops and resizes
# them and stores the data in a numpy.ndarray object
def read_input_maps(path, image_name, suffix, amount_maps = 10, crop_pixels = 20, output_shape = [128, 128, 3],
                    progress_step = 0):
    """
    Function that reads amount of satellite images, crops and resizes them and stores them in a numpy.ndarray which is
    readable for Keras.

    :param path:
    :param image_name:
    :param suffix:
    :param amount_maps:
    :param crop_pixels:
    :param output_shape:
    :param progress_step:
    :return:

    :param path: path of the satellite maps.
    :param image_name: prefix of each map in the path.
    :param suffix: suffix of each map in the path.
    :param amount_maps: amount of map that should be read in. Has to match the argument from read_label_masks.
    :param crop_pixels: cropping option. Not used currently.
    :param output_shape: output dimensions of the data. We used 512x512 as dimension for our model.
    :param progress_step: controls after how many loaded images the user wants to be informed.
    :return: numpy.ndarray of label masks.
    """


    map_list = []
    for i in range(amount_maps):
        map = skimage.io.imread(fname = (path + image_name + str(i) + suffix), as_grey = False)
        # print(map.shape)
        map = crop_and_resize_maps(input=map, crop_pixels = crop_pixels, output_shape = output_shape)
        if progress_step != 0:
            if i % progress_step  == 0:
                print("Read Input Map: ", i)
        map_list.append(map)

    map_list = np.vstack(map_list)
    map_list = map_list.reshape(-1, output_shape[0], output_shape[1], output_shape[2])
    # Jann Trial to get uint8 satellites
    # map_list = np.ndarray.astype(map_list, 'uint8')

    return map_list


# ---------------------------------



def read_label_masks(path, mask_name, suffix, amount_masks=10, crop_pixels=20, output_shape=[128, 128],
                    progress_step = 0, separation = False):

    """
    Function that reads amount of masks, crops and resizes them and stores them in a numpy.ndarray which is readable for
    Keras.

    :param path: path of the label masks.
    :param mask_name: prefix of each mask in the path.
    :param suffix: suffix of each mask in the path.
    :param amount_masks: amount of masks that should be read in. Has to match the argument from read_input_maps.
    :param crop_pixels: cropping option. Not used currently.
    :param output_shape: output dimensions of the data. We used 512x512 as dimension for our model.
    :param progress_step: controls after how many loaded images the user wants to be informed.
    :param separation: not used, to be deleted soon.
    :return: numpy.ndarray of label masks.
    """
    # standard creation of a list of masks depth = 1
    if separation == False:
        mask_list = []
        for i in range(amount_masks):
            mask = skimage.io.imread(fname=(path + mask_name + str(i) + suffix), as_grey=False)
            # use crop_and_resize_MASKS here!!!
            mask = crop_and_resize_masks(input=mask, crop_pixels=crop_pixels, output_shape=output_shape)
            if progress_step != 0:
                if i % progress_step  == 0:
                    print("Read Label Mask: ", i)
            mask_list.append(mask)

        mask_list = np.vstack(mask_list)
        mask_list = mask_list.reshape(-1, output_shape[0], output_shape[1])

        # lambda function sets all given values = 1 if it is original 0 (=black) and 0 if not (= white / gray)
        # threshold can be adjusted later for inclusion of gray-road-areas
        thresholder = lambda x: 1 if x == 0 else 0
        # vectorization makes computing more efficient
        vfunc = np.vectorize(thresholder)
        mask_list = vfunc(mask_list)

        # change data type to float64 as required by tflearn
        mask_list = np.ndarray.astype(mask_list, 'float32')

        return mask_list

    # creation of list of label masks with depth = 2 for the "special tflearn case"
    elif separation == True:

        mask_list_1 = []

        for i in range(amount_masks):
            mask = skimage.io.imread(fname=(path + mask_name + str(i) + suffix), as_grey=False)
            # use crop_and_resize_MASKS here!!!
            mask = crop_and_resize_masks(input=mask, crop_pixels=crop_pixels, output_shape=output_shape)
            if progress_step != 0:
                if i % progress_step == 0:
                    print("Read Label Mask: ", i, " in stacking mode")
            mask_list_1.append(mask)

        mask_list_1 = np.vstack(mask_list_1)
        mask_list_1 = mask_list_1.reshape(-1, output_shape[0], output_shape[1], 1)

        # lambda function sets all given values = 1 if it is original 0 (=black) and 0 if not (= white / gray)
        # threshold can be adjusted later for inclusion of gray-road-areas
        thresholder = lambda x: 1 if x == 0 else 0
        # vectorization makes computing more efficient
        vfunc = np.vectorize(thresholder)
        mask_list_1 = vfunc(mask_list_1)

        # function that assigns opposite entries in each cell (0 => 1, 1 => 0) for label mask with depth 2
        reverser = lambda x: 1 if x == 0 else 0
        reverse_func = np.vectorize(reverser)
        mask_list_2 = reverse_func(mask_list_1)

        # stacked along depth
        stackedMask = np.concatenate((mask_list_1, mask_list_2), axis=3)
        # change data type to float64 as required by tflearn
        stackedMask = np.ndarray.astype(stackedMask, 'float32')

        return stackedMask

# ---------------------------------

def sampler(path_orig, path_sampled, amountsamples):
    """
    Function that randomly samples amount X of images from folder of data. Takes as Input: folder path with
    (already preprocessed) satellite and road imagery and returns: new folder with X randomly sampled image pairs
    in subsequent order (0,....,X)
    :param path_orig: original path
    :param path_sampled: new path for sampled images
    :param amountsamples: integer value for desired amount of sampled image sets
    :return:
    """
    # seed
    random.seed(3337)
    # get amount of image pairs in orig path
    amountFiles = int(len([f for f in os.listdir(path_orig) if os.path.isfile(os.path.join(path_orig, f))]) * 0.5)
    # randomly sample indices from range of all files
    indices = random.sample(range(amountFiles), amountsamples)

    # set counter for sequential storage
    counter = 0
    # loop over index list, load and save imagery
    for i in indices:
        sat = Image.open((path_orig + 'crop_sat' + str(i) + '.png'))
        lab = Image.open((path_orig + 'crop_lab' + str(i) + '.png'))

        sat.save((path_sampled + 'train_sat' + str(counter) + '.png'))
        lab.save((path_sampled + 'train_lab' + str(counter) + '.png'))

        counter += 1#
    print('Done! I sampled ' + str(counter), 'Image Couples')

# ---------------------------------


######################################################################
#
#   EVALUATION
#
######################################################################



# --------------------------------

def threshold_predictions(prediction, threshold = 0.5):
    """
    Function that threshold predictions in [0,1] to hard values in {0,1}.
    Input: prediction as numpy.ndarray, threshold in [0,1]
    Output: <class 'numpy.ndarray'>, dimension (variable): (128, 128), format: int64 , values max, min: 1 0

    :param prediction: prediction as numpy array
    :param threshold: a certain threshold after which the pixel is classified as road
    :return: the thresholded prediction as a numpy.ndarray
    """

    thresholder = lambda x: 1 if x >= (1-threshold) else 0
    # vectorization makes computing more efficient
    vfunc = np.vectorize(thresholder)

    thresh_pred = vfunc(prediction)
    return thresh_pred

# --------------------------------

# CLASS RESULTS
#
# Class that calculates many measures for a given hard-thresholded prediction as outputed by threshold_prediction()
# and a ndarray with the correct labels also given as a numpy.ndarray with format int64 and values in {0,1}.
# This can be solved more intelligently but the class serves our need in this format
#
# Input: prediction and true ndarray
# Output: object with many attributes including the most important measures, the resultmatrix and the confustion-matrix
#       as a python dictionary
#
# IMPORTANT HOW-TO:
#   1) res1 = result(prediction=p_d_orig, truevalue=y_d_orig)   înstantiate thhe
#   2) res1.calculate_measures()                                CALL THIS FUNCTION at the beginning, fills class with values
#   3) res1.printmeasures()
#   4) use attributes such as res1.acc
#
#
# IMPORTANT: Currently labels are labelled as: road=0, non-road=1 (the 'other way round')
#           TODO: should be adjusted someday and then everything has to be updated to road=1, non-road=1
#           this affects also MAINLY the computation of the TP/FP/TN/FN values

class result:

    def __init__(self, prediction, truevalue):
        # we read in a numpy ndarray and need to flatten it to some sort of a list (not exactly)
        self.Yhat = prediction.flatten('C')
        self.Y = truevalue.flatten('C')
        self.confusionlist = []
        self.countdict = dict
        self.resultmatrix = np.empty(shape=prediction.shape)
        self.tpr = 0
        self.prec = 0
        self.acc = 0
        self.f1 = 0
        self.fpr = 0
        self.neg = 0
        self.pos = 0
        self.summary = str

    def calculate_measures(self):

        #### based on confusion list
        # initialize empty list for storage of TP/FP/FN/TN
        confusion_list = []
        # loop over both flattened arrays using zip
        # TODO: currently switched labels: road=0, non-road=1. Thus switched confusion matrix

        self.neg = (self.Y == 1).sum()
        self.pos = (self.Y == 0).sum()


        for i, j in zip(self.Yhat, self.Y):
            if i == 0 and j == 0:
                confusion_list.append('TP')
            elif i == 1 and j == 1:
                confusion_list.append('TN')
            elif i == 1 and j == 0:
                confusion_list.append('FN')
            elif i == 0 and j == 1:
                confusion_list.append('FP')
        self.confusionlist = confusion_list

        ## calculate draw matrix
        # get the TP/FN/FP/TN Matrix in shape of the input image
        self.resultmatrix = np.asarray(self.confusionlist).reshape(self.resultmatrix.shape)


        ## calculate Measures
        # get counts dictionary
        self.countdict = collections.Counter(self.confusionlist)

        # TPR = RECALL
        # control for 0 division
        if (self.countdict['TP'] + self.countdict['TN']) > 0:
            self.tpr = self.countdict['TP'] / (self.countdict['TP'] + self.countdict['FN'])

        # PRECISION
        # control for 0 division
        if (self.countdict['TP'] + self.countdict['FP']) > 0:
            self.prec = self.countdict['TP'] / (self.countdict['TP'] + self.countdict['FP'])

        # F1 = harmonic mean precision and recall
        # control for 0 division
        if (self.prec + self.tpr) > 0:
            self.f1 = 2 * (self.prec * self.tpr) / (self.prec + self.tpr)

        # ACCURACY
        self.acc = (self.countdict['TP'] + self.countdict['TN']) / len(confusion_list)

        # FALSE POSITIVE RATE FOR ROC CURVES
        # control for 0 division
        if (self.countdict['TP'] + self.countdict['FP']) > 0:
            self.fpr = 1- self.countdict['TN'] / (self.countdict['TN'] + self.countdict['FP'])

        # SUMMARY
        self.summary = ('Recall: ', round(self.tpr,2), 'Precision: ', round(self.prec,2), 'Accuracy: ', round(self.acc, 2),
                        'F1: ', round(self.f1, 2), 'Specificity: ', round(self.fpr, 2), 'Posratio:', round(self.pos / (self.neg + self.pos)),2)

    def printmeasures(self):
        print('Recall: ', round(self.tpr,2), 'Precision: ', round(self.prec,2), 'Accuracy: ', round(self.acc, 2), 'F1: ',
              round(self.f1, 2),'Specificity: ', round(self.fpr, 2), 'Positive ratio:', round(self.pos / (self.neg + self.pos)) )


# --------------------------------

def visualize_prediction(resultmatrix, x_image):
    """
    Function that draws TP/FN/FP from prediction on input satellite imagery for visualization of the prediction
    Volodymir & Mnih - Style. Input: result_matrix containing the Confusion matrix values for each pixel which is the
    Output from class result.resultmatrix and satellite image array alread transformed to numpy.ndarray.
    How to: (1) Image.open(path), then 2) np.asarray())
    Outputs a x_image which is a numpy.ndarray with RGB values: matrix with depth 3 containing of uint8 values in
    {0, ..., 255}. Can be easily plotted via Image.fromarray(x_image, mode ='RGB').show() which is used in the test.py.

    IMPORTANT: Currently labels are labelled as: road=0, non-road=1 (the 'other way round')
    TODO: should be adjusted someday and then everything has to be updated to road=1, non-road=1. This affects also
    MAINLY the computation of the TP/FP/TN/FN values

    :param resultmatrix: output of class result
    :param x_image: the original satellite image as a numpy.ndarray
    :return: vizualization of the pixelwise predictions as a numpy.ndarray
    """

    # flag image as writeable
    x_image.flags.writeable = True

    for i in range(resultmatrix.shape[0]):
        for j in range(resultmatrix.shape[0]):
            measure = resultmatrix[i, j]
            # check for all 4 cases and draw coloured point accordingly
            if measure == 'TP':
                # green
                x_image[i, j, 0] = 255
                x_image[i, j, 1] = 215
                x_image[i, j, 2] = 0
            elif measure == 'FN':
                # blue
                x_image[i, j, 0] = 0
                x_image[i, j, 1] = 0
                x_image[i, j, 2] = 255
                # red
            elif measure == 'FP':
                x_image[i, j, 0] = 255
                x_image[i, j, 1] = 0
                x_image[i, j, 2] = 0

    return x_image



# ---------------------------------
#   VISUALIZATION FUNCTIONS WITH PILLOW aka PIL.Image
#TODO: maybe merge to one function in case of massive boredom


def print_two_images(image1, image2, background = (255, 200, 255), textcolor = (100, 100, 100), header = 'header', title1 = 'image 1',
                     title2 = 'image 2', storagepath = 'None', plotit = True):
    """
    Prints two already read in PIL.Image-objects next to each other in one plot and adapts to the size of the images
    (assuming quadratic and equal-sized images). Use rgb values such from
    https://www.w3schools.com/colors/colors_picker.asp to set text and background color. The User can choose to print
    and/ or store the resulting concatenated image (default: only plot, no storage). The storage format is .png RGB and
    can only be adapted in this source code, no user interaction. The user can annotate with text with the given
    defaults. Requires PIL with Image and ImageDraw.

    This is implemented in a version for 2, 3, 4 images and in a quadrativ version. Can be merged in one elegant
    function one day.

    This is the version with 2 images.

    :param image1: path to image 1
    :param image2: path to image 2
    :param background: RGB background color
    :param textcolor: RGB textcolor
    :param header: string given by user.
    :param title1: string given by user.
    :param title2: string given by user.
    :param storagepath: path for the storage of the images. If non is given, nothing is stored
    :param plotit: plot them or not, boolean.
    :return: combined graphic of the input images.
    """
    # store height, needed for later creation of the plot.
    # quadratic images assumed
    height = image1.size[0]

    # initialize new image. We paste the two real images in this one
    new_concat = Image.new(mode='RGB', size=(2 * height + 30, 40 + height), color= background)
    # give paste the upper left corner as input
    # paste(image, (x, y)), where (0,0) is in the upper left corner
    new_concat.paste(image1, (10, 30))
    new_concat.paste(image2, (20 + height, 30))

    # draw text in the image
    draw_all = ImageDraw.Draw(new_concat)
    draw_all.text(xy=(10, 0), text=header, fill = textcolor)
    draw_all.text(xy=(10, 20), text=title1, fill = textcolor)
    draw_all.text(xy=(20 + height, 20), text=title2, fill = textcolor)

    # plot the image if required. Ubuntu will use imagemagick for this by default
    if plotit:
        new_concat.show()

    # option to store the image and not only plot it
    if storagepath != 'None':
        new_concat.save(storagepath, 'PNG')



def print_three_images(image1, image2, image3, background = (255, 200, 255), textcolor = (100, 100, 100),
                       header = 'header', title1 = 'image 1', title2 = 'image 2', title3 = 'image 3',
                       storagepath = 'None', plotit = True):
    """
    Prints three already read in PIL.Image-objects next to each other in one plot and adapts to the size of the images
    (assuming quadratic and equal-sized images). Use rgb values such from
    https://www.w3schools.com/colors/colors_picker.asp to set text and background color. The User can choose to print
    and/ or store the resulting concatenated image (default: only plot, no storage). The storage format is .png RGB and
    can only be adapted in this source code, no user interaction. The user can annotate with text with the given
    defaults. Requires PIL with Image and ImageDraw.

    This is implemented in a version for 2, 3, 4 images and in a quadrativ version. Can be merged in one elegant
    function one day.

    This is the version with 3 images.

    :param image1: path to image 1
    :param image2: path to image 2#
    :param image3: path to image 3
    :param background: RGB background color
    :param textcolor: RGB textcolor
    :param header: string given by user.
    :param title1: string given by user.
    :param title2: string given by user.
    :param storagepath: path for the storage of the images. If non is given, nothing is stored
    :param plotit: plot them or not, boolean.
    :return: combined graphic of the input images.
    """
    # store height, needed for later creation of the plot.
    # quadratic images assumed
    height = image1.size[0]
    # print(height)

    # initialize new image. We paste the three real images in this one
    new_concat = Image.new(mode='RGB', size=(3 * height + 10 + 3*10, 40 + height), color= background)
    # give paste the upper left corner as input
    # paste(image, (x, y)), where (0,0) is in the upper left corner
    new_concat.paste(image1, (10, 30))
    new_concat.paste(image2, (20 + height, 30))
    new_concat.paste(image3, (30 + 2*height, 30))


    # draw text in the image
    draw_all = ImageDraw.Draw(new_concat)
    draw_all.text(xy=(10, 0), text=header, fill = textcolor)
    draw_all.text(xy=(10, 20), text=title1, fill = textcolor)
    draw_all.text(xy=(20 + height, 20), text=title2, fill = textcolor)
    draw_all.text(xy=(30 + 2*height, 20), text=title3, fill = textcolor)


    # plot the image if required. Ubuntu will use imagemagick for this by default
    if plotit:
        new_concat.show()

    # option to store the image and not only plot it
    if storagepath != 'None':
        new_concat.save(storagepath, 'PNG')


def print_four_images(image1, image2, image3, image4, background=(255, 200, 255), textcolor=(100, 100, 100),
                      header='header', title1='image 1', title2='image 2', title3='image 3', title4='image4',
                      subtitle = 'subtitle', storagepath='None', plotit=True):
    """
    Prints four already read in PIL.Image-objects next to each other in one plot and adapts to the size of the images
    (assuming quadratic and equal-sized images). Use rgb values such from
    https://www.w3schools.com/colors/colors_picker.asp to set text and background color. The User can choose to print
    and/ or store the resulting concatenated image (default: only plot, no storage). The storage format is .png RGB and
    can only be adapted in this source code, no user interaction. The user can annotate with text with the given
    defaults. Requires PIL with Image and ImageDraw.

    This is implemented in a version for 2, 3, 4 images and in a quadrativ version. Can be merged in one elegant
    function one day.

    This is the version with 3 images.

    :param image1: path to image 1
    :param image2: path to image 2
    :param image3: path to image 3
    :param image4: path to image 4
    :param background: RGB background color
    :param textcolor: RGB textcolor
    :param header: string given by user.
    :param title1: string given by user.
    :param title2: string given by user.
    :param storagepath: path for the storage of the images. If non is given, nothing is stored
    :param plotit: plot them or not, boolean.
    :return: combined graphic of the input images.
    """

    # store height, needed for later creation of the plot.
    # quadratic images assumed
    height = image1.size[0]
    # print(height)

    # initialize new image. We paste the three real images in this one
    new_concat = Image.new(mode='RGB', size=(4 * height + 10 + 4 * 10, 60 + height), color=background)
    # give paste the upper left corner as input
    # paste(image, (x, y)), where (0,0) is in the upper left corner
    new_concat.paste(image1, (10, 30))
    new_concat.paste(image2, (20 + height, 30))
    new_concat.paste(image3, (30 + 2 * height, 30))
    new_concat.paste(image4, (40 + 3 * height, 30))

    # draw text in the image
    draw_all = ImageDraw.Draw(new_concat)
    draw_all.text(xy=(10, 0), text=header, fill=textcolor)
    draw_all.text(xy=(10, 20), text=title1, fill=textcolor)
    draw_all.text(xy=(20 + height, 20), text=title2, fill=textcolor)
    draw_all.text(xy=(30 + 2 * height, 20), text=title3, fill=textcolor)
    draw_all.text(xy=(40 + 3 * height, 20), text=title4, fill=textcolor)
    draw_all.text(xy=(10, 40 + height), text=subtitle, fill=textcolor)


    # plot the image if required. Ubuntu will use imagemagick for this by default
    if plotit:
        new_concat.show()

    # option to store the image and not only plot it
    if storagepath != 'None':
        new_concat.save(storagepath, 'PNG')

def print_four_images_quadratic(image1, image2, image3, image4, background=(255, 200, 255), textcolor=(100, 100, 100),
                         header='header', title1='image 1', title2='image 2', title3='image 3', title4='image4',
                         subtitle='subtitle', storagepath='None', plotit=True):
    """
    Prints four already read in PIL.Image-objects in a quadratic way in one plot and adapts to the size of the images
    (assuming quadratic and equal-sized images). Use rgb values such from
    https://www.w3schools.com/colors/colors_picker.asp to set text and background color. The User can choose to print
    and/ or store the resulting concatenated image (default: only plot, no storage). The storage format is .png RGB and
    can only be adapted in this source code, no user interaction. The user can annotate with text with the given
    defaults. Requires PIL with Image and ImageDraw.

    This is implemented in a version for 2, 3, 4 images and in a quadrativ version. Can be merged in one elegant
    function one day.

    This is the version with 3 images.

    :param image1: path to image 1
    :param image2: path to image 2
    :param image3: path to image 3
    :param image4: path to image 4
    :param background: RGB background color
    :param textcolor: RGB textcolor
    :param header: string given by user.
    :param title1: string given by user.
    :param title2: string given by user.
    :param storagepath: path for the storage of the images. If non is given, nothing is stored
    :param plotit: plot them or not, boolean.
    :return: combined graphic of the input images.
        """
      # store height, needed for later creation of the plot.
     # quadratic images assumed
    height = image1.size[0]
        # print(height)

     # initialize new image. We paste the three real images in this one
    new_concat = Image.new(mode='RGB', size=(2 * height + 30, 90 + 2* height), color=background)
     # give paste the upper left corner as input
     # paste(image, (x, y)), where (0,0) is in the upper left corner
    new_concat.paste(image1, (10, 30))
    new_concat.paste(image2, (20 + height, 30))
    new_concat.paste(image3, (10, 60 + height))
    new_concat.paste(image4, (20 + height, 60 + height))

     # draw text in the image
    draw_all = ImageDraw.Draw(new_concat)
    draw_all.text(xy=(10, 0), text=header, fill=textcolor)
    draw_all.text(xy=(10, 20), text=title1, fill=textcolor)
    draw_all.text(xy=(20 + height, 20), text=title2, fill=textcolor)
    draw_all.text(xy=(10, 50 + height), text=title3, fill=textcolor)
    draw_all.text(xy=(20 + height, 50 + height), text=title4, fill=textcolor)
    draw_all.text(xy=(10, 70 + 2*height), text=subtitle, fill=textcolor)

    # plot the image if required. Ubuntu will use imagemagick for this by default
    if plotit:
        new_concat.show()

     # option to store the image and not only plot it
    if storagepath != 'None':
        new_concat.save(storagepath, 'PNG')



######################################################################
#
#   HELPER FUNCTIONS that are currently not in use
#
######################################################################


# Function that crops and resizes a list of input maps (satellites) given in np.ndarray format

def crop_and_resize_maps(input, crop_pixels = 20, output_shape = [128, 128, 3]):
    # crop the input map over all three dimensions
    temp_input = input[crop_pixels:input.shape[0]-crop_pixels, crop_pixels:input.shape[1]-crop_pixels, : ]
    # resize the input map
    temp_input = skimage.transform.resize(image = temp_input, output_shape = output_shape)
    return temp_input

# Function that crops and resizes a list of label-masks (roads) given in np.ndarray format
# differs from the map cropper because of the dimensions
# TODO: include both versions in one generic function later

# ---------------------------------

def crop_and_resize_masks(input, crop_pixels = 20, output_shape = [128, 128]):
    # crop the input map over all three dimensions
    temp_input = input[crop_pixels:input.shape[0]-crop_pixels, crop_pixels:input.shape[1]-crop_pixels]
    # resize the input map
    temp_input = skimage.transform.resize(image = temp_input, output_shape = output_shape)
    return temp_input


# DEMO SCRIPT FOR POSTPROCESSING OF ONE PREDICTION
"""
import custommodule
import numpy as np
from PIL import Image

y = np.load(file = '/home/jgucci/PycharmProjects/consulting_master/foo_data/y31.npy')
y = np.ndarray.astype(y, 'int64')
p = np.load(file = '/home/jgucci/PycharmProjects/consulting_master/foo_data/p31.npy')

p_thresh = custommodule.threshold_predictions(p, threshold=0.8)

res1 = custommodule.result(prediction=p_thresh, truevalue=y)
res1.calculate_measures()
print(res1.summary)

sat = Image.open('/home/jgucci/PycharmProjects/consulting_master/foo_data/crop_sat31.png')
satarray = np.asarray(sat)

vis1 = custommodule.visualize_prediction(resultmatrix=res1.resultmatrix, x_image=satarray)

# re-pic the arrays
vis1_repic = Image.fromarray(vis1)
vis1_repic.show()

y_repic = np.ndarray.astype(y*255, dtype='uint8')
y_repic = Image.fromarray(y_repic, mode='L')

p_thresh_repic = np.ndarray.astype(p_thresh*255, dtype='uint8')
p_thresh_repic = Image.fromarray(p_thresh_repic, mode ='L')

# print three images in one pillow image
custommodule.print_three_images(header=str(res1.summary),image1=y_repic, image2=p_thresh_repic, image3=vis1_repic, title1='true',
                                title2='predicted', title3='coloured satellite', plotit=True)
"""

