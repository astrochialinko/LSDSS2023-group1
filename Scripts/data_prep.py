import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import glob
import pickle
import os
from sklearn.model_selection import train_test_split

PATH_DATA_DIR = '../Data/'  # path for the data directory
IMSIZE_SPLIT = 64           # the size of the split image

def read_tif(save_pickle=True):
    """
    Read the tif files and save them as the pickle files (dict)

    parameters:
        save_pickle: save the pickle files if True (bool)

    outputs:
        None
    """

    # path for the filenames
    dir_path = PATH_DATA_DIR + '**/*.tif'
    filepaths = glob.glob(dir_path, recursive=True) # list
    filepaths.sort()

    filenames = []
    labels = []
    features = []

    num_images = len(filepaths)
    num_row, num_col, num_channel = np.array(Image.open(filepaths[0])).shape
    images = np.zeros((num_images, num_row, num_col, num_channel))

    for i, filepath in enumerate(filepaths):
        
        # get the filename, label, and feature
        filename = filepath.split('/')[-1].split('.')[0] # e.g., sj-03-2810_001
        label    = filepath.split('/')[-2]               # e.g., CLL
        feature  = filepath.split('/')[-1].split('_')[0] # e.g., sj-03-2810
        
        filenames.append(filename)
        labels.append(label)
        features.append(feature)
        
        # read the tif file
        im = Image.open(filepath)
        imarray = np.array(im)
        images[i] = imarray

    # save to the dictory
    lymphoma_dict = {'images': images,
                     'labels': np.array(labels),
                     'filenames': np.array(filenames),
                     'features': np.array(features)
                    }

    if save_pickle:
        # save the dictory as the pickel files
        with open(PATH_DATA_DIR+'lymphoma.pickle', 'wb') as file:
            pickle.dump(lymphoma_dict, file)

    return lymphoma_dict

def trim_and_split_data(save_npy=True, data_path = PATH_DATA_DIR):
    """
    Trim and split the images and save them as the npy files (numpy array)

    parameters:
        save_pickle: save the npy files if True (bool)

    outputs:
        None
    """

    # Use loads to load the variable
    with open(PATH_DATA_DIR+'lymphoma.pickle', 'rb') as file:
        lymphoma_data = pickle.load(file)
    print(f'Read the pickel file that with the keys: {list(lymphoma_data.keys())}')

    # get the images
    images = lymphoma_data['images']
    num_images, num_rows, num_cols, num_channels = images.shape
    print(f'Original image shape:{ images.shape}')

    # trim the image
    imsize_trim  = 1024   # the size of the trimmed image
    im_trim_startpx = 10  # the start pixel to trim the image
    image_trim = images[:, 
                        im_trim_startpx:im_trim_startpx+imsize_trim, 
                        im_trim_startpx:im_trim_startpx+imsize_trim, 
                        :]
    print(f'Trim the image to the shape of: {image_trim.shape}')

    # Split the image to many subimages
    image_split = np.array([image_trim[i][x:x+IMSIZE_SPLIT, y:y+IMSIZE_SPLIT] 
                            for i in range(num_images) 
                            for x in range(0, imsize_trim, IMSIZE_SPLIT) 
                            for y in range(0, imsize_trim, IMSIZE_SPLIT)])
    print(f'Split the image to the shape of: {image_split.shape}')


    # match the lables, filenames, features to the split images
    labels    = lymphoma_data['labels']

    # the number of the split subimages in the original trim image
    num_subimg_in_orgimg = (imsize_trim/IMSIZE_SPLIT)**2 

    labels_split = np.repeat(labels, num_subimg_in_orgimg)

    if save_npy:
        # save the numpy array as the npy files
        with open(data_path+'X_data.npy', 'wb') as file:
            np.save(file, image_split)
            print(f'save the X_data.npy to {data_path}X_data.npy')

        with open(data_path+'y_data.npy', 'wb') as file:
            np.save(file, labels_split)
            print(f'save the y_data.npy to {data_path}y_data.npy')

def make_train_test(imfile, labelfile, data_path=PATH_DATA_DIR):
    """
    Process the data in the pickle files into train/test/val sets
    Save as numpy files
    """

    # Get data from pickles
    with open(imfile, 'rb') as filename:
        images = np.load(filename)
    with open(labelfile, 'rb') as filename:
        labels = np.load(filename)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1)

    # Save the npy files
    with open(data_path+'X_train.npy', 'wb') as file:
        np.save(file, X_train)
        print(f'save the X_train.npy to {data_path}X_train.npy')
    with open(data_path+'y_train.npy', 'wb') as file:
        np.save(file, y_train)
        print(f'save the y_train.npy to {data_path}y_train.npy')
    with open(data_path+'X_test.npy', 'wb') as file:
        np.save(file, X_test)
        print(f'save the X_test.npy to {data_path}X_test.npy')
    with open(data_path+'y_test.npy', 'wb') as file:
        np.save(file, y_test)
        print(f'save the y_test.npy to {data_path}y_test.npy')

def main():
    read_tif(save_pickle=True)             # 12.9 GB
    trim_and_split_data(save_npy=True)     # 9.5 GB
    make_train_test(PATH_DATA_DIR+'X_data.npy', PATH_DATA_DIR+'y_data.npy')


main()