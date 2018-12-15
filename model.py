import numpy as np
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
%matplotlib inline

dir_path = "carnd"
cols = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
df = pd.read_csv(dir_path + "/driving_log.csv", names=cols, )

def get_correct_file_path(x):
    return x.split('/')[-1]

df['center'] = df['center'].apply(get_correct_file_path)
df['right'] = df['right'].apply(get_correct_file_path)
df['left'] = df['left'].apply(get_correct_file_path)

bin_threshold = 200
remove_ixs = []
for i in range(total_bins):
    temp_list = []
    for j in range(df.steering.size):
        if df['steering'][j] >= bins[i] and df['steering'][j] <= bins[i+1]:
        temp_list.append(j)
    # ensure that data is dropped randomly rather than a specific portion of the recording
    temp_list = shuffle(temp_list)
    temp_list = temp_list[bin_threshold:]
    remove_ixs.extend(temp_list)
print("Before: ", len(df))
df.drop(df.index[remove_ixs], inplace=True)
print("After: ", len(df))

imgdir = 'carnd/IMG/'
def load_data(df):
    path = []
    measurement = []
    for i in range(len(df)):
        ixs = df.iloc[i]
        center, left, right = ixs[0], ixs[1], ixs[2]
        path.append(imgdir + center.strip())
        path.append(imgdir + left.strip())
        path.append(imgdir + right.strip())
        measurement.append(float(ixs[3]))
    paths = np.asarray(path)
    measurements = np.asarray(measurement)
    return paths, measurements

paths, measurements = load_data(df)
print(paths.shape, measurements.shape)

X_train, X_valid, y_train, y_valid = train_test_split(paths, measurements, test_size=0.2, random_state=42)
print(X_train.shape, X_valid.shape)

def zoom(img):
    zoomed = iaa.Affine(scale=(1,1.3))
    return zoomed.augment_image(img)

def pan(img):
    panned = iaa.Affine(translate_percent={'x': (-0.1,0.1), 'y': (-0.1,0.1)})
    return panned.augment_image(img)

def alter_brightness(img):
    iaa.Multiply((0.2, 1.2))

def augment_data(img):
    img = zoom(img)

def preprocess(img):

    # convert to YUV space as Nvidia Model we will use has used the same space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # slice the unwanted sections from the image
    img = img[60:135,:,:]
    # smooth the image using a gaussian kernel: denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # reshape to fit the input size of Nvidia Model
    img = cv2.resize(img, (200, 66))
    # normalize
    img = img/255

    return img

X_train = np.array(list(map(preprocess, X_train)))
X_valid = np.array(list(map(preprocess, X_valid)))

print(X_train.shape, X_valid.shape)
#Model
def NvidiaModel():

    # defining our model
    model = Sequential()

    # 1 conv2D layer=> input: 66x200x3, total_params: 5x5x24x3+24=1824
    model.add(Conv2D(24, (5, 5), strides=(2,2), input_shape=(66, 200, 3), activation='relu'))

    # 2 conv2D layer=> input: 31x98x24, total_params: 5x5x36x24+36=21636
    model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))

    # 3 conv2D layer=> input: 14x47x36, total_params: 5x5x48x36+48=43248
    model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))

    # 4 conv2D layer=> input: 5x22x48, total_params=3x3x64x48+64=27712
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # 5 conv2D layer=> input: 3x20x64, total_params: 3x3x64x64+64=36928, output_shape=(1x18x64)
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Dropout=> total_params=0, output_shape=1x18x64
    model.add(Dropout(0.5))

    # Flatten=> total_params=0, output_shape: 1152
    model.add(Flatten())

    # Dense=> output_shape=100, total_params=1152x100+100=115300
    model.add(Dense(100, activation='relu'))

    # Dropout=> total_params=0, output_shape=100
    model.add(Dropout(0.5))

    # Dense=> total_params=100x50+50=5050
    model.add(Dense(50, activation='relu'))

    # Dense=> total_params=50x10+10=510
    model.add(Dense(10, activation='relu'))

    # Dense=> total_params=10x1+1=11
    model.add(Dense(1))

    # compile
    optimizer=Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model

model = NvidiaModel()
# model.summary()

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=15, batch_size=64, shuffle=1)
# model.save("model.h5")