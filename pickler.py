import os
import pickle

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

img_dir = 'raw-pacs/pacs_data/pacs_data/'  # Change this to unpack PACS from different directories.
output_dir = 'materials/pacs/'  # Change this to pickle PACS to different directories.

# Variables
classes_dict = {"dog": 1, "elephant": 2, "giraffe": 3, "guitar": 4, "horse": 5, "house": 6, "person": 7, }
domains_list = ["photo", "sketch", "cartoon", "art_painting"]
test_domain = "cartoon"
output_name = "PACS"
image_size = (32, 32)
test_size = 0.1

# Initialize empty lists
data_test = []
labels_test = []
data_rest = []
labels_rest = []

print("Start pickling classes")

for class_ in classes_dict:
    if domains_list is None: domains_list = [""]  # Use no domain if no domains list is specified
    for domain in domains_list:
        for filename in os.listdir(img_dir + domain + "/" + class_):
            if filename.endswith((".jpg", ".png")):
                image = Image.open(img_dir + domain + "/" + class_ + "/" + filename)
                image = image.resize(image_size, Image.ANTIALIAS)
                if domain == test_domain:
                    data_test.append(np.asarray(image))
                    labels_test.append(classes_dict[class_])
                else:
                    data_rest.append(np.asarray(image))
                    labels_rest.append(classes_dict[class_])

data_train, data_val, labels_train, labels_val = train_test_split(data_rest, labels_rest, test_size=test_size)

# Pickle the dicts
train_dict = {"data": data_train, "labels": labels_train, }
pickle.dump(train_dict, open(output_dir + output_name + "_train.pickle", "wb"))

test_dict = {"data": data_test, "labels": labels_test, }
pickle.dump(test_dict, open(output_dir + output_name + "_test.pickle", "wb"))

val_dict = {"data": data_val, "labels": labels_val, }
pickle.dump(val_dict, open(output_dir + output_name + "_val.pickle", "wb"))

print("Pickling finished")
