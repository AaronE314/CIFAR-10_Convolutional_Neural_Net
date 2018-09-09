import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class Conv_Net:

    def __init__(self, class_names, num_predictions=20, epochs=25, batch_size=32, data_aug=False):
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.weights_dir = os.path.join(os.getcwd(), 'saved_models')
        self.pdf_dir = os.path.join(os.getcwd(), 'output')
        self.model_name = 'cifar10.h5'
        self.num_predictions = num_predictions
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_augmentation=data_aug


        self._load_data()
        self._create_model()
        self._load_weights()

    def _load_data(self):
        # Load data
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = cifar10.load_data()

        # Normalize
        self.train_images = self.train_images.astype('float32')
        self.test_images = self.test_images.astype('float32')
        self.train_images /= 255
        self.test_images /= 255

        # One hot encode
        self.train_labels = to_categorical(self.train_labels, self.num_classes)
        self.test_labels = to_categorical(self.test_labels, self.num_classes)

    def _create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(96, (3,3), input_shape=self.train_images.shape[1:], activation='relu'))
        self.model.add(Conv2D(96, (3,3), strides=2, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(192, (3,3), activation='relu'))
        self.model.add(Conv2D(192, (3,3), strides=2, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self):

        if not self.data_augmentation:
            print('Not using data augmentation')
            self.model.fit(self.train_images, self.train_labels, 
                validation_data=(self.test_images, self.test_labels), 
                epochs=self.epochs, 
                batch_size=self.batch_size,
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

            datagen.fit(self.train_images)

            self.model.fit_generator(datagen.flow(self.train_images, self.train_labels,
                                                batch_size=self.batch_size),
                                    epochs=self.epochs,
                                    validation_data=(self.test_images, self.test_labels),
                                    workers=4,
                                    shuffle=True)

        self._save_weights()
        self.evaluate()

    def evaluate(self):
        # Final evaluation of the model
        metrics = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
                
        print('Test loss: {:.4f}'.format(metrics[0]))
        print('Test accuracy: {:.2f}%'.format(metrics[1] * 100))

    def predict(self, val):
        return self.model.predict(val)

    def _load_weights(self):
        if not os.path.isdir(self.weights_dir):
            print("No Weights found.")
            return
        model_path = os.path.join(self.weights_dir, self.model_name)
        self.model.load_weights(model_path)

    def _save_weights(self):

        # Save model and weights
        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)
        model_path = os.path.join(self.weights_dir, self.model_name)
        self.model.save(model_path)
        print('Saved trained model at {:s}'.format(model_path))

    def _plot_image(self, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], np.argmax(true_label[i]), img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    
        plt.imshow(img, interpolation='spline16')

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
    
        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    self.class_names[true_label]),
                                    color=color)

    def _plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], np.argmax(true_label[i])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1]) 
        predicted_label = np.argmax(predictions_array)
        
        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def plot_results(self, num_rows, num_cols):

        predictions = self.model.predict(self.test_images)
        num_images = num_rows*num_cols

        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            self._plot_image(i, predictions, self.test_labels, self.test_images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            self._plot_value_array(i, predictions, self.test_labels)
        plt.show()

    def save_results_as_pdf(self):
        
        if not os.path.isdir(self.pdf_dir):
            os.makedirs(self.pdf_dir)
        pdf_path = os.path.join(self.pdf_dir, 'predictions.pdf')
        pdf = PdfPages(pdf_path)

        num_rows = 4
        num_cols = 2
        predictions = self.model.predict(self.test_images)
        num_images = num_rows*num_cols

        #total_images = len(self.test_images)
        total_images = 160

        for j in range(0, total_images, num_images):
            chunk_img = self.test_images[j:j + num_images]
            chunk_labels = self.test_labels[j:j + num_images]
            fig = plt.figure(figsize=(2*2*num_cols, 2*num_rows))
            for i in range(num_images):
                plt.subplot(num_rows, 2*num_cols, 2*i+1)
                self._plot_image(i, predictions, chunk_labels, chunk_img)
                plt.subplot(num_rows, 2*num_cols, 2*i+2)
                self._plot_value_array(i, predictions, chunk_labels)
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()