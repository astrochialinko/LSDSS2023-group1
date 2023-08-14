import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import glob
import pickle
from sklearn.model_selection import train_test_split
import os

PATH_DATA_DIR = '../Data/'  # path for the data directory
PATH_PICKLE = '../Data/'    # path for the pickle file
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

    print('Reading tiff files')

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

    print('Saving to pickle file')

    if save_pickle:
        # save the dictory as the pickel files
        with open(PATH_PICKLE+'lymphoma.pickle', 'wb') as file:
            pickle.dump(lymphoma_dict, file)
    
    print('Saved pickle')

    return lymphoma_dict

def trim_and_split_data(save_pickle=True):
    """
    Trim and split the images and save them as the pickle files (numpy array)

    parameters:
        save_pickle: save the pickle files if True (bool)

    outputs:
        None
    """

    print('Loading pickle file')

    # Use loads to load the variable
    with open(PATH_PICKLE+'lymphoma.pickle', 'rb') as file:
        lymphoma_data = pickle.load(file)
    print(f'Read the pickel file that with the keys: {list(lymphoma_data.keys())}')

    # get the images
    images = lymphoma_data['images']
    num_images, num_rows, num_cols, num_channels = images.shape
    print(f'Original image shape:{ images.shape}')

    print('Trimming')

    # trim the image
    imsize_trim  = 1024   # the size of the trimmed image
    im_trim_startpx = 10  # the start pixel to trim the image
    image_trim = images[:, 
                        im_trim_startpx:im_trim_startpx+imsize_trim, 
                        im_trim_startpx:im_trim_startpx+imsize_trim, 
                        :]
    print(f'Trim the image to the shape of: {image_trim.shape}')

    print('Splitting')

    # Split the image to many subimages
    image_split = np.array([image_trim[i][x:x+IMSIZE_SPLIT, y:y+IMSIZE_SPLIT] 
                            for i in range(num_images) 
                            for x in range(0, imsize_trim, IMSIZE_SPLIT) 
                            for y in range(0, imsize_trim, IMSIZE_SPLIT)])
    print(f'Split the image to the shape of: {image_split.shape}')

    # match the lables, filenames, features to the split images
    labels    = lymphoma_data['labels']
    filenames = lymphoma_data['filenames']
    features  = lymphoma_data['features']

    # the number of the split subimages in the original trim image
    num_subimg_in_orgimg = (imsize_trim/IMSIZE_SPLIT)**2 

    labels_split = np.repeat(labels, num_subimg_in_orgimg)
    filenames_split = np.repeat(filenames, num_subimg_in_orgimg)
    features_split = np.repeat(features, num_subimg_in_orgimg)

    print('Saving trimed pickles')

    if save_pickle:
        # save the numpy array as the pickel files
        with open(PATH_PICKLE+'X_data.pickle', 'wb') as file:
            pickle.dump(image_split, file)
            print(f'save the X_data.pickle to {PATH_PICKLE}X_data.pickle')

        with open(PATH_PICKLE+'y_data.pickle', 'wb') as file:
            pickle.dump(labels_split, file)
            print(f'save the y_data.pickle to {PATH_PICKLE}y_data.pickle')

        with open(PATH_PICKLE+'X_filename.pickle', 'wb') as file:
            pickle.dump(filenames_split, file)
            print(f'save the X_filename.pickle to {PATH_PICKLE}X_filename.pickle')

        with open(PATH_PICKLE+'X_feature.pickle', 'wb') as file:
            pickle.dump(features_split, file)
            print(f'save the X_feature.pickle to {PATH_PICKLE}X_feature.pickle')


def load_data_from_pickles(imfile, labelfile):
    '''
    Process the data in the pickle files into train/test/val sets
    Save as numpy files
    '''
    print('Reading in split X and split y pickles')
    # Get data from pickles
    with open(imfile, 'rb') as filename:
            images = pickle.load(filename)
    with open(labelfile, 'rb') as filename:
            labels = pickle.load(filename)
    print('Splitting into training and testing npy files')
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1)
    np.save(PATH_DATA_DIR+'X_train.npy', X_train)
    np.save(PATH_DATA_DIR+'y_train.npy', y_train)
    np.save(PATH_DATA_DIR+'X_test.npy', X_test)
    np.save(PATH_DATA_DIR+'y_test.npy', y_test)
     

def main():
    # read_tif(save_pickle=True)             # 12.9 GB
    # trim_and_split_data(save_pickle=True)  # 9.5 GB
    load_data_from_pickles(PATH_PICKLE+'X_data.pickle', PATH_PICKLE+'y_data.pickle')

main()