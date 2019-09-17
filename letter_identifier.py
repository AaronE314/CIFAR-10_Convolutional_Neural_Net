
from emnist import extract_training_samples, list_datasets
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Letter_Conv_Net:

    def __init__(self, saveName = 'letter_conv_net', num_predictions=20, epochs=25, batch_size=32, data_aug=False, color=False):

        self.class_names = [chr(ord('A') + i) for i in range(26)]
        self.class_names.insert(0, 'A')
        self.num_classes = len(self.class_names)
        self.weights_dir = os.path.join(os.getcwd(), 'saved_models')
        self.pdf_dir = os.path.join(os.getcwd(), 'output')
        self.model_name = '{}.h5'.format(saveName)
        self.num_predictions = num_predictions
        self.saveName = saveName
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_augmentation=data_aug
        self.color = 3 if color else 1
        
        self._load_data()
        self._create_model()
        self._load_weights()

    def _load_data(self):

        print('loading data')
        
        X, y = extract_training_samples('letters')

        self.train_images, self.test_images = X[:60000], X[60000:70000]
        self.train_labels, self.test_labels = y[:60000], y[60000:70000]

        self.train_images = self.train_images.astype('float32')
        self.test_images = self.test_images.astype('float32')
        self.train_images /= 255
        self.test_images /= 255

        self.org_images = self.test_images
        self.train_images = self.train_images.reshape(60000, 28, 28, 1)
        self.test_images = self.test_images.reshape(10000, 28, 28, 1)

        # One hot encode
        self.train_labels = to_categorical(self.train_labels, self.num_classes)
        self.test_labels = to_categorical(self.test_labels, self.num_classes)

    def _create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (5,5), input_shape=self.train_images.shape[1:], activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(32, (3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(32, (3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
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
        model_path = os.path.join(self.weights_dir, self.model_name)
        if not os.path.isdir(self.weights_dir) or not os.path.isfile(model_path):
            print("No Weights found.")
            return
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
    
        plt.imshow(img, cmap=plt.cm.get_cmap('binary'))

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
        thisplot = plt.bar(range(27), predictions_array, color="#777777")
        plt.xticks(range(27), self.class_names)
        plt.tick_params(labelsize=4)
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
            self._plot_image(i, predictions, self.test_labels, self.org_images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            self._plot_value_array(i, predictions, self.test_labels)
        plt.show()

    def save_results_as_pdf(self):
        
        if not os.path.isdir(self.pdf_dir):
            os.makedirs(self.pdf_dir)
        pdf_path = os.path.join(self.pdf_dir, '{}.pdf'.format(self.saveName))
        pdf = PdfPages(pdf_path)

        num_rows = 4
        num_cols = 2
        predictions = self.model.predict(self.test_images)
        num_images = num_rows*num_cols

        total_images = len(self.test_images)
        # total_images = 160

        for j in range(0, total_images, num_images):
            chunk_img = self.org_images[j:j + num_images]
            chunk_labels = self.test_labels[j:j + num_images]
            chunk_predictions = predictions[j:j + num_images]
            fig = plt.figure(figsize=(2*2*num_cols, 2*num_rows))
            for i in range(num_images):
                plt.subplot(num_rows, 2*num_cols, 2*i+1)
                self._plot_image(i, chunk_predictions, chunk_labels, chunk_img)
                plt.subplot(num_rows, 2*num_cols, 2*i+2)
                self._plot_value_array(i, chunk_predictions, chunk_labels)
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()

def main():
    convNet = Letter_Conv_Net()

    if len(sys.argv) > 1 and sys.argv[1] == 1:
        convNet.train()

    convNet.plot_results(5, 5)
    # convNet.save_results_as_pdf()


if __name__ == '__main__':
    main()