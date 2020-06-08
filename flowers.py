import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

flowers_path = "./flowers"


X = []
labels = []

for flower_dir in os.listdir(flowers_path):
	flower_name = os.fsdecode(flower_dir)
	print(f'{flower_name}:')
	flower_name_path = f'{flowers_path}/{flower_name}'
	for flower in tqdm(os.listdir(flower_name_path)):
		current_flower_path = f'{flower_name_path}/{flower}'
		try:
			img = Image.open(current_flower_path)
			rgb_img = np.array(img.resize((250, 250)))

			X.append(rgb_img)
			labels.append(str(flower_name))
		except Exception as e:
			print(e)
			continue

		


X = np.array(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y, 5)

# preproccesing
X = X/255

# spliting to train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# building the model
model = models.Sequential()

# Conv+Pooling 1
model.add(layers.Conv2D(filters=64, kernel_size=(3,3) ,input_shape=(250,250,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv+Pooling 2
model.add(layers.Conv2D(filters=32, kernel_size=(3,3) ,activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(layers.Flatten())

# Fully connected layers
model.add(layers.Dense(units=16, activation='relu'))
model.add(layers.Dense(units=5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("learning...")

model.fit(X_train, y_train, batch_size=2, epochs=10, validation_data=(X_test, y_test))

print(model)