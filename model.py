import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from tqdm import tqdm
from cv2 import cv2
from sklearn.utils import shuffle

os.listdir("/home/xxx/python/mask/data")

class_names = ["mask", "no_mask"]
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}  # nummerieren
img_size = (100, 100)

def load():
    paths = [
        "/home/xxx/python/mask/data/train",
        "/home/xxx/python/mask/data/test",
    ]
    output = []

    for dataset in paths:

        images = []
        labels = []

        print("Loading {}".format(dataset))

        for x in os.listdir(dataset):
            label = class_names_label[x]

            for file in tqdm(os.listdir(os.path.join(dataset, x))):
                img_path = os.path.join(os.path.join(dataset, x), file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, img_size)

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype="float32") / 255.0
        labels = np.array(labels, dtype="int32")

        output.append((images, labels))

    return output

# pylint: disable=unbalanced-tuple-unpacking
(
    (train_images, train_labels),
    (test_images, test_labels),
) = load()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

