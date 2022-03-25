import numpy as np
from PIL import Image
import pickle
import os
from sklearn.model_selection import train_test_split

img_dir = 'raw-pacs/pacs_data/pacs_data/photo/'
classes_dict = {"dog": 1, "elephant": 2, "giraffe": 3, "guitar": 4, "horse": 5, "house": 6, "person": 7, }
data = []
labels = []

print("Start pickling classes")
for class_ in classes_dict:
    for filename in os.listdir(img_dir + class_):
        if filename.endswith((".jpg", ".png")):
            image = Image.open(img_dir + class_ + "/" + filename)
            image = image.resize((32, 32), Image.ANTIALIAS)
            data.append(np.asarray(image))
            labels.append(classes_dict[class_])

data_train, data_rest, labels_train, labels_rest = train_test_split(data, labels, test_size=0.30)
data_val, data_test, labels_val, labels_test = train_test_split(data_rest, labels_rest, test_size=0.50)

# Pickle the dicts
train_dict = {"data": data_train, "labels": labels_train, }
pickle.dump(train_dict, open("materials/pacs/PACS_train.pickle", "wb"))

test_dict = {"data": data_test, "labels": labels_test, }
pickle.dump(test_dict, open("materials/pacs/PACS_test.pickle", "wb"))

val_dict = {"data": data_val, "labels": labels_val, }
pickle.dump(val_dict, open("materials/pacs/PACS_val.pickle", "wb"))

print("Pickling finished")
