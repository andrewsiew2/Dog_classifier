'''
Classification of 6 classes from Stanford Dog Dataset for UW CSE455 Final Proj
http://vision.stanford.edu/aditya86/ImageNetDogs/main.html

Classes:
  Chihuahua
  Maltese_dog
  papillon
  Japanese_spaniel
  Blenheim_spaniel
  beagle
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from random import shuffle
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
from keras.callbacks import Callback
import pickle
import scipy.io

'''
Basic:
    Flip_LR = False
    Flip_UD = False
    DROPOUT = False
    BATCH_NORM = FALSE


Flip
'''

RUN_NAME = "CNN_5L"

### TESTING VARIABLE SETTINGS ###
NUM_TRAIN = None
NUM_TEST = None
FLIP_LR = False
FLIP_UD = False
DROPOUT = False
BATCH_NORM = True


### DEFAULT SETTINGS ###
IMG_SIZE = 300
DIR_LISTS = ""
DIR_TEST_DATA = "test.txt"
DIR_TRAIN_DATA = "train.txt"
DIR_LABELS = "file_list.mat"
DIR_CLASSES = "class_list.txt"
NUM_SPECIES = 6
# Training settings
BATCH_SIZE = 10
EPOCHS = 20

def main():
    print("Starting up engines...")
    train_data = load_training_data(max_len=NUM_TRAIN, fliplr=FLIP_LR, flipud=FLIP_UD)
    print("Training data len: %d" % len(train_data))
    test_data = load_test_data(max_len=NUM_TEST, fliplr=FLIP_LR, flipud=FLIP_UD)
    print("Testing data len: %d" % len(test_data))

    print("Reshaping Images and Labels...")
    trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    trainLabels = np.array([i[1] for i in train_data])
    testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    testLabels = np.array([i[1] for i in test_data])

    model = make_model()

    print("Compiling Model...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


    print("Training Model...")
    history = model.fit(trainImages, trainLabels, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1, validation_data=(testImages, testLabels))
    #https://keras.io/models/model/
    loss, acc = model.evaluate(testImages, testLabels, verbose = 5)
    plot_history(history)
    save_history(history)
    # print("Accuracy: %f" % (acc * 100))


def make_model():
    # Simple CNN
    print("Assembling Model..")
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if BATCH_NORM: model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if BATCH_NORM: model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if BATCH_NORM: model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if BATCH_NORM: model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if BATCH_NORM: model.add(BatchNormalization())
    if DROPOUT: model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if DROPOUT: model.add(Dropout(0.3))
    model.add(Dense(NUM_SPECIES, activation = 'softmax'))
    return model


# I didn't talk about this one in the poster because 
# i experiemented with it at the last second
def make_3_2Layer_CNN_model():
    # Simple CNN
    print("Assembling Model..")
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if BATCH_NORM: model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if BATCH_NORM: model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if BATCH_NORM: model.add(BatchNormalization())
    if DROPOUT: model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    if DROPOUT: model.(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    if DROPOUT: model.add(Dropout(0.3))
    model.add(Dense(NUM_SPECIES, activation = 'softmax'))
    return model

def make_two_layer_CNN_model():
    # Simple CNN
    print("Assembling Model..")
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if BATCH_NORM: model.add(BatchNormalization())
    if DROPOUT: model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    if DROPOUT: model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    if DROPOUT: model.add(Dropout(0.3))
    model.add(Dense(NUM_SPECIES, activation = 'softmax'))
    return model

def make_one_layer_model():
    print("Assembling Model..")
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    if DROPOUT: model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    if DROPOUT: model.add(Dropout(0.3))
    model.add(Dense(NUM_SPECIES, activation='relu'))
    return model


def save_history(history):
    filename = 'plots/history/%s_History_Epochs_%d_%d%d%d%d.pickle' % (RUN_NAME, EPOCHS, int(FLIP_LR), int(FLIP_UD), int(DROPOUT), int(BATCH_NORM))
    with open(filename, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def plot_history(history):
    # print(history.history.keys())
    # "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Train/Test Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('plots/accuracy/%s_Accuracy_Epochs_%d_%d%d%d%d.png' % (RUN_NAME, EPOCHS, int(FLIP_LR), int(FLIP_UD), int(DROPOUT), int(BATCH_NORM)))
    #plt.show()
    plt.gcf().clear()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Train/Test Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('plots/loss/%s_Loss_Epochs_%d_%d%d%d%d.png' % (RUN_NAME, EPOCHS, int(FLIP_LR), int(FLIP_UD), int(DROPOUT), int(BATCH_NORM)))
    #plt.show()
    plt.gcf().clear()


#returns the label of an image
def make_one_hot(num):
    ret = np.zeros(NUM_SPECIES)
    ret[num] = 1
    return ret


### DATA LOADING METHODS ###
# loads training data
def load_training_data(max_len=None, fliplr=None, flipud=None):
    return load_data(DIR_LISTS + DIR_TRAIN_DATA, max_len=max_len, fliplr=fliplr, flipud=flipud)


# loads testing data
def load_test_data(max_len=None, fliplr=None, flipud=None):
    return load_data(DIR_LISTS + DIR_TEST_DATA, max_len=max_len, fliplr=fliplr, flipud=flipud)


# loads data from given source location
def load_data(location, max_len=None, fliplr=None, flipud=None):
    data = []
    f = open(location)
    fileContents = f.read().strip("\n")
    fileContents = fileContents.split('\n')
    if not max_len:
        max_len = len(fileContents)-1
    for i in range(max_len):
        words = fileContents[i].split("  ")
        path = "Images/%s" % words[0]
        label = make_one_hot(int(words[1]))
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        data.append([np.array(img), label])

    	# Basic Data Augmentation - Horizontal Flipping
        if fliplr or flipud:
            flip_img = Image.open(path)
            flip_img = flip_img.convert('L')
            flip_img = flip_img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            flip_img = np.array(flip_img)
            if fliplr:
                fliplr_img = np.fliplr(flip_img)
                data.append([fliplr_img, label])
            if flipud:
                flipud_img = np.flipud(flip_img)
                data.append([flipud_img, label])
            if flipud and fliplr:
                flipudlr_img = np.fliplr(np.flipud(flip_img))
                data.append([flipudlr_img, label])
    shuffle(data)
    return data

if __name__ == "__main__":
    main()
