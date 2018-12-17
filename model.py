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

# read the csv file for preprocessing
dir_path = "carnd"
cols = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
df = pd.read_csv(dir_path + "/driving_log.csv", names=cols, )

def get_correct_file_path(x):
    '''
    Converts the file paths by removing the local directory paths that can be
    used for processing on any workspace
    '''
    return x.split('/')[-1]

df['center'] = df['center'].apply(get_correct_file_path)
df['right'] = df['right'].apply(get_correct_file_path)
df['left'] = df['left'].apply(get_correct_file_path)

# Threshold by 200 images per bin to remove highly biased images belonging to
# steering angle 0 bins
total_bins = 21
hist, bins = np.histogram(df['steering'], total_bins)
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

# prepare the features and labels
# paths has the path of each centered image
# measurements has the corresponding steering angle measurement for the images
imgdir = 'carnd/IMG/'
def load_data(df):
    path = []
    measurement = []
    for i in range(len(df)):
        ixs = df.iloc[i]
        center, left, right = ixs[0], ixs[1], ixs[2]
        path.append(imgdir + center.strip())
        # path.append(imgdir + left.strip())
        # path.append(imgdir + right.strip())
        measurement.append(float(ixs[3]))
    paths = np.asarray(path)
    measurements = np.asarray(measurement)
    return paths, measurements

paths, measurements = load_data(df)
print(paths.shape, measurements.shape)

# split the dataset into train and validation set as a measure to reduce overfitting
X_train, X_valid, y_train, y_valid = train_test_split(paths, measurements, test_size=0.2, random_state=42)
print(X_train.shape, X_valid.shape)

def preprocess(img):
    '''
    Preprocesses the image by
    - using the YUV channel as used in the Nvidia model
    - choosing the region of interest
    - introducing a gaussian smoothing function to denoise the image
    - resize to fit the Nvidia model 200x66x3 input image shape
    - normalize
    '''

    # convert to YUV space as Nvidia Model we will use has used the same space
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2YUV)
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

# Image augmentation
# NOTE: Augmentation has not been used in the submission (v1)
def zoom(img):
  zoomed = iaa.Affine(scale=(1,1.3))
  return zoomed.augment_image(img)

def pan(img):
  panned = iaa.Affine(translate_percent={'x': (-0.1,0.1), 'y': (-0.1,0.1)})
  return panned.augment_image(img)

def alter_brightness(img):
  b = iaa.Multiply((0.2, 1.2))
  return b.augment_image(img)

def flip_image(img, steering_ang):
  img = cv2.flip(img, 1)
  return img, -steering_ang

def augment_data(img, ang):
  if np.random.rand() < 0.5:
    img = zoom(img)
  if np.random.rand() < 0.5:
    img = pan(img)
  if np.random.rand() < 0.5:
    img = alter_brightness(img)
  if np.random.rand() < 0.5:
    img = flip_image(img)
  return img, ang

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

# intiialize the model and vie
model = NvidiaModel()
model.summary()

# train the model
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=15, batch_size=64, shuffle=1)

# save the trained model which can be used to run the drive.py file
model.save("model.h5")

# viewing the plots for accuracy and loss
# Loss
plt.plot(history.history['loss']);
plt.plot(history.history['val_loss']);
plt.legend(['Training','Validation']);
plt.title('Loss');
plt.xlabel('Epoch');

# Accuracy
plt.plot(history.history['acc']);
plt.plot(history.history['val_acc']);
plt.legend(['Training','Validation']);
plt.title('Accuracy');
plt.xlabel('Epoch');
