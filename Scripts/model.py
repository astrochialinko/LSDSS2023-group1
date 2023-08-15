import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import itertools

PATH_DATA_DIR = '../Data/'               # path for the data directory
PATH_OUTPUT_DIR = '../Output/'           # path for the output directory

def load_data(data_path=PATH_DATA_DIR):
    """
    Load the train and test data from the npy files
    """
    # Get data from npy
    with open(data_path+'X_train.npy', 'rb') as filename:
        X_train= np.load(filename)
    with open(data_path+'y_train.npy', 'rb') as filename:
        y_train_word= np.load(filename)
    with open(data_path+'X_test.npy', 'rb') as filename:
        X_test= np.load(filename)
    with open(data_path+'y_test.npy', 'rb') as filename:
        y_test_word= np.load(filename)

    return X_train, y_train_word, X_test, y_test_word

def plot_first_36(X_train, y_train_word, data_path=PATH_OUTPUT_DIR):
    """
    plot first 36 images in Lymphoma
    """
    fig, ax = plt.subplots(6, 6, figsize = (12, 12))
    fig.suptitle('First 36 images in lymphoma')
    fig.tight_layout(pad = 0.1, rect = [0, 0, 0.93, 0.93])
    for x, y in [(i, j) for i in range(6) for j in range(6)]:
        ax[x, y].imshow(X_train[x + y * 6].reshape((64, 64, 3)).astype(np.uint8))
        ax[x, y].set_title(y_train_word[x + y * 6])
    plt.savefig(data_path+'first_36_lymphoma.pdf', dpi = 300)

def normailize_data(X_train, X_test):
    """
    Normalize the data
    """
    X_train /= 255
    X_test /= 255

    return X_train, X_test

def transform_label(y_train_word, y_test_word):
    """
    Transform the label from string to numerical
    """
    # CLL -> 0; FL -> 1; MCL -> 2

    le = preprocessing.LabelEncoder()
    le.fit(y_train_word)
    y_train = le.transform(y_train_word)

    le = preprocessing.LabelEncoder()
    le.fit(y_test_word)
    y_test = le.transform(y_test_word)

    print('There are', X_train.shape[0], 'training data and', X_test.shape[0], 'testing data')
    print('Number of occurence for each number in training data (0 stands for 10):')
    print(np.vstack((np.unique(y_train), np.bincount(y_train))).T)

    return y_train, y_test, le


def create_validation_set(X_train, y_train, X_test, y_test):
    """
    Create the validation dataset
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=0.2, shuffle = True, 
                                                      random_state= 8, stratify=y_train)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def one_hot_encoding(y_train, y_val, y_test):
    """
    transform training label to one-hot encoding
    """
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = lb.transform(y_train)
    Y_val = lb.transform(y_val)
    Y_test = lb.transform(y_test)
    print("Shape after one-hot encoding: ", Y_train.shape)

    return Y_train, Y_val, Y_test, lb

def create_model(img_size, n_classes):

    print('Building model...')
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape = img_size, kernel_initializer = 'normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (5, 5), kernel_initializer = 'normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model

def model_fit(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs):
    history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val))
    print('Validating model...')
    score, acc = model.evaluate(X_val, Y_val, verbose = 1)
    print('\nLoss:', score, '\nAcc:', acc)
    model.save('model.h5')

def model_predict(model, X_test, y_test, lb):
    
    # Predict
    print('Predicting...')
    Y_prob_pred = model.predict(X_test)
    Y_pred = lb.inverse_transform(Y_prob_pred)
    Y_pred_2d = [[y] for y in Y_pred]
    index = [[i] for i in range(1, X_test.shape[0] + 1)]

    output_np = np.concatenate((index, Y_pred_2d), axis = 1)
    output_df = pd.DataFrame(data = output_np, columns = ['ImageId', 'Label'])
    output_df.to_csv('out.csv', index = False)

    print(
        f"Classification report:\n"
        f"{metrics.classification_report(y_test, Y_pred)}\n"
    )

    return Y_pred

def cal_confusion_matrix(y_test, Y_pred):
    cm = confusion_matrix(y_test, Y_pred)
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    print(cm)

    return cm

def plot_confusion_matrix(cm, n_classes, fig_path=PATH_OUTPUT_DIR):
    # plot the cm
    plt.imshow(cm, cmap = 'gray')
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, np.arange(n_classes), rotation=45)
    plt.yticks(tick_marks, np.arange(n_classes))

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fig_path + 'cm.pdf', dpi = 300)

def main():

    # data preparation
    X_train, y_train_word, X_test, y_test_word = load_data
    plot_first_36(X_train, y_train_word)
    X_train, X_test = normailize_data(X_train, X_test)
    y_train, y_test, le = transform_label(y_train_word, y_test_word)
    X_train, X_val, X_test, y_train, y_val, y_test = create_validation_set(X_train, y_train, X_test, y_test)
    Y_train, Y_val, Y_test, lb = one_hot_encoding(y_train, y_val, y_test)

    # model
    img_size = (64, 64, 3)
    n_classes = 3
    epochs = 10
    model = create_model(img_size=img_size, n_classes=n_classes)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    model = model_fit(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs)
    Y_pred = model_predict(model, X_test, y_test, lb)
    
    # evaluation
    cm = cal_confusion_matrix(y_test, Y_pred)
    plot_confusion_matrix(cm, n_classes)

main()