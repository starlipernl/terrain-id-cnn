"""
Author: Nathan Starliper
This script fine tunes a pretrained Inceptionv3 model trained on imagenet. We will be fine tuning the top layer of
the model to use for prediction of our terrain classes.

"""

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Data augmentation
# loading dataset from pickle files (created from load_images.py)
with open('features.pkl', 'rb') as file:
    features = pickle.load(file)
with open('labels.pkl', 'rb') as file:
    labels = pickle.load(file)
feature_list = np.asarray(features)
label_list = np.asarray(labels).reshape((-1, 1))
# 80/20 training testing split
x_train, x_test, y_train, y_test = train_test_split(feature_list, label_list, test_size=0.2)
y_test = y_test.reshape((len(y_test),))
x_train = x_train.astype('float32') / 255.0  # normalize [0,1]
x_test = x_test.astype('float32') / 255.0
# further split training data to 80/20 train validation split
val_split = round(x_train.shape[0] * 0.8)
x_valid = x_train[val_split:]
y_valid = y_train[val_split:]
x_train = x_train[0:val_split]
y_train = y_train[0:val_split]
# initialize data augmentation
datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=20,
                             horizontal_flip=True, width_shift_range=0.15, height_shift_range=0.15)
val_datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False)
datagen.fit(x_train)
val_datagen.fit(x_valid)

# HYPERPARAMETERS AND DESIGN CHOICES, if reoptimized, update these values from the values recorded in the
# hyperparameters csv created by run_hyperopt_cnn.py
NUM_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.00075633  # use learning rate 1/10 of the rate we used for our cnn
NUM_NEURONS_IN_DENSE_1 = 256
DROP_PROB = 0.02076

# image input
input_tensor = Input(shape=(224, 224, 3))
# load Inception model pretrained in imagenet without top layer (we will add new one for our application)
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a 6 class output layer
predictions = Dense(6, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# freeze all layers except the top layer we will be fine tuning
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',  metrics=["accuracy"])


# callbacks for Save weights, Tensorboard
# creating a new directory for each run using timestamp
# folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
# history_file = folder + "/inception" + ".h5"
# save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
# tb_callback = TensorBoard(log_dir=folder)
# train model with data generator
train_accuracy, train_loss, valid_accuracy, valid_loss = cnn.train_model_datagen(model, datagen, val_datagen, x_train,
                                                         y_train, BATCH_SIZE, NUM_EPOCHS, x_valid, y_valid)
# test model
test_accuracy, test_loss, predictions = cnn.test_model(model, x_test, y_test)

# Visualization and Output
num_epochs_plot = range(1, len(train_accuracy) + 1)  # x axis range
# Loss curves
plt.figure()
plt.plot(num_epochs_plot, train_loss, "b", label="Training Loss")
plt.plot(num_epochs_plot, valid_loss, "r", label="Validation Loss")
plt.title("Loss Curves")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Figures/loss_inception.png')
plt.close()
#plt.show()

# Accuracy curves
plt.figure(2)
plt.plot(num_epochs_plot, train_accuracy, "b", label="Training Accuracy")
plt.plot(num_epochs_plot, valid_accuracy, "r", label="Validation Accuracy")
plt.title("Accuracy Curves")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Figures/acc_inception.png')
plt.close()
#plt.show()

# Test loss and accuracy
print("\n##########")
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(test_accuracy))
