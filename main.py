import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Helper libraries
import matplotlib.pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10.h5'
num_predictions = 20
epochs = 25
batch_size = 32
data_augmentation=False

# Load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# # Reshape
# train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 3).astype('float32')
# test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 3).astype('float32')

# Normalize
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# One hot encode
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)


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
# model = Sequential()
# model.add(Conv2D(32, (3,3), padding='same', input_shape=train_images.shape[1:], activation='relu'))
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

model = Sequential()
model.add(Conv2D(96, (3,3), input_shape=train_images.shape[1:], activation='relu'))
model.add(Conv2D(96, (3,3), strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(192, (3,3), activation='relu'))
model.add(Conv2D(192, (3,3), strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# # init RMprop opt
# opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation')
    model.fit(train_images, train_labels, 
          validation_data=(test_images, test_labels), 
          epochs=epochs, 
          batch_size=batch_size,
          shuffle=True)
else:
    print('Using data augmentation')

    # Will preprocess the images before fitting
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
    )

    datagen.fit(train_images)

    model.fit_generator(datagen.flow(train_images, train_labels,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(test_images, test_labels),
                        workers=4,
                        shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at {:s}'.format(model_path))

# Final evaluation of the model
metrics = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss: {:.4f}'.format(metrics[0]))
print('Test accuracy: {:.2f}%'.format(metrics[1] * 100))