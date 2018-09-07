import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam


# Helper libraries
import matplotlib.pyplot as plt

# Import dataset
cifar = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar.load_data()

# Reshape
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 3).astype('float32')

# Reshape
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 3).astype('float32')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# # Normalize
# train_images /= 255
# test_images /= 255

# # Show first 25 scaled images with there label
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i][0]])

# plt.show()

# model
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(train_images.shape[1], train_images.shape[2], 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Train the layers
model.fit(train_images, train_labels, 
          validation_data=(test_images, test_labels), 
          epochs=10, 
          batch_size=200,
          shuffle=True)

# Save model
model.save('cifar10.h5')

# Final evaluation of the model
metrics = model.evaluate(test_images, test_labels, verbose=0)
print("Metrics (Test loss & Test Accuracy):")
print(metrics)