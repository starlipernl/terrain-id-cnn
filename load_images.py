"""
Code used for image preprocessing to rescale images to the size specified by the dim parameter. To extract images as
color images set the color variable to 1, otherwise set the variable to 0 for grayscale.
"""

import os
import cv2
import pickle
import numpy as np

# names of folders where images are stored. Each directory corresponds to a label.
class_folder = ['Asphalt', 'Carpet', 'Cobblestone', 'Grass', 'Mulch', 'Tile']
class_id = -1
feature_list = []
label_list = []
color = 1  # set color option: 0 - grayscale, 1 - color
dim = 64  # set rescale size, image resize to dim x dim
for c in class_folder:
    class_id += 1
    path = 'Data/' + c + '/'
    image_filenames = [os.path.join(path, file) for file in os.listdir(path) if file.startswith('Frame')]
    for filename in image_filenames:
        if color == 1:
            img = cv2.imread(filename, 1)  # imread color option
        else:
            img = cv2.imread(filename, 0)
        img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
        if color == 0:
            img = np.expand_dims(img, axis=2)
        feature_list.append(img)
        label_list.append(class_id)
# save lists to pickle files
with open('features.pkl', 'wb') as file:
    pickle.dump(feature_list, file)
with open('labels.pkl', 'wb') as file:
    pickle.dump(label_list, file)