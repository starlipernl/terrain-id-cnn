import os
import cv2
import pickle
import numpy as np

class_folder = ['Asphalt', 'Carpet', 'Cobblestone', 'Grass', 'Mulch', 'Tile']
class_id = -1
feature_list = []
label_list = []
color = 0
for c in class_folder:
    class_id += 1
    path = 'Data/' + c + '/'
    image_filenames = [os.path.join(path, file) for file in os.listdir(path) if file.startswith('Frame')]
    for filename in image_filenames:
        if color == 1:
            img = cv2.imread(filename, 1)
        else:
            img = cv2.imread(filename, 0)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        if color ==0:
            img = np.expand_dims(img, axis=2)
        feature_list.append(img)
        label_list.append(class_id)
with open('features.pkl', 'wb') as file:
    pickle.dump(feature_list, file)
with open('labels.pkl', 'wb') as file:
    pickle.dump(label_list, file)