from conv_net_mnist import Conv_Net
from emnist import extract_training_samples

x, y = extract_training_samples('letters')


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

mnist_cnn = Conv_Net(class_names, fashion_mnist, 'fashion_mnist', color=False, epochs=15)

mnist_cnn.train()

#mnist_cnn.plot_results(5, 5)

#mnist_cnn.evaluate()

mnist_cnn.save_results_as_pdf()