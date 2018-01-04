'''
Preprocessing Script

reads in the satellites and the maps in 512x512 pixel format and processes them to a format that is readable by the
train script. This script is also used for dataset augmentation using the augment-option. If set to TRUE,
one could also use the thresholder to use only "positive enough" images that do contain at least X% of street

Input: set of raw images from mnih

Output: cropped and augmented patches stored in subfolder "preprocessed". Ready to use for the
modeling and testing part.

'''

import custommodule
import os
custommodule.checker()

# read mnih data, preprocess and store in new folder "preprocessed"
storage = "/home/jgucci/Desktop/mnih_data/"

# iterate over all 3 dataset
datasets = ['train', 'valid', 'test']

for set in datasets:

    storagefolder = storage + "/" + set + "/preprocessed/"

    if not os.path.exists(storagefolder):
        os.makedirs(storagefolder)

    # amount of images in this folder:
    orig_folder = storage + "/" + set + "/raw/"
    amount_inputs = int(len([f for f in os.listdir(orig_folder) if os.path.isfile(os.path.join(orig_folder, f))]) * 0.5)


    # path to raw tiff data
    raw_path = storage + "/" + set + "/raw/"

    print("\n" + "processing " + set + " images containing " + str(amount_inputs) + " files" +  "\n")

    custommodule.read_split_save_many(satellite = 'sat', label = 'map',
                                        images_path = raw_path,
                                        store_path = storagefolder,
                                        size_crop = 512, threshold = 0.05, amount_inputs = amount_inputs,
                                        augment = False, thresholdgreen= 0, usetiffs=True)






