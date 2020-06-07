from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
# making the data:

X = []
labels = []
dataset_path = "/home/nivgold/Projects/Flower_Classifier/FlowersDataset"
dataset_dir = os.fsencode(dataset_path)
for flower_type in os.listdir(dataset_dir):
    flower_name = os.fsdecode(flower_type)
    current_flower_path = os.path.join(dataset_path, flower_name)
    current_flower_dir = os.fsencode(current_flower_path)
    print(flower_name+":")
    for instance in os.listdir(current_flower_dir):
        image_name = os.fsdecode(instance)
        img_path = os.path.join(current_flower_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        try:
            img = cv2.resize(img, (150,150))
            X.append(np.array(img))
            labels.append(flower_name)
        except Exception as e:
            print(e)
            continue



labels = np.array(labels)
X = np.array(X)
X = X/255

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, shuffle=True)

# building the model
model = models.Sequential()
model.add(layers.Conv2D(64, (3,3), input_shape=(150,150,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

