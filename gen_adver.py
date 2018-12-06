from scipy.misc import imsave
from keras import metrics
from PIL import Image

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.utils.np_utils import to_categorical

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import cnn


# import data
x_train, y_train, x_test, y_test = cnn.data()

# HYPERPARAMETERS AND DESIGN CHOICES
NUM_EPOCHS = 50
BATCH_SIZE = 20
LEARNING_RATE = 0.0001
NUM_NEURONS_IN_DENSE_1 = 128
DROP_PROB = 0.5
ACTIV_FN = "relu"
activation_fn = cnn.get_activ_fn(ACTIV_FN)
num_classes = 6
input_shape = np.shape(x_train[1])
model = cnn.build_model(input_shape, activation_fn, LEARNING_RATE, DROP_PROB, NUM_NEURONS_IN_DENSE_1, num_classes)
model.load_weights('cnn_relu.h5')
test_img = np.expand_dims(x_train[1], 0)
target_idx = model.predict(test_img).argmax()
target = to_categorical(target_idx, 6)
target_variable = K.variable(target)
loss = metrics.categorical_crossentropy(model.output, target_variable)
gradients = K.gradients(loss, model.input)
get_grad_values = K.function([model.input], gradients)
grad_values = get_grad_values([test_img])[0]
grad_signs = np.sign(grad_values)
epsilon = 0.1
perturbation = test_img * 0.1
modified_array = test_img + perturbation
new_img = test_img, 0., 1.

print(model.predict(new_img))
