import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

flowers_path = "./flowers"

#print(os.listdir(flowers_path))

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
			labels.append(flower_name)
		except Exception as e:
			print(e)
			continue

		


X = np.array(X)
labels = np.array(labels)
print(f'labels shape: {labels.shape}')

# preproccesing
X = X/255

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, shuffle=True)

print(X_train.shape)
print(y_train.shape)