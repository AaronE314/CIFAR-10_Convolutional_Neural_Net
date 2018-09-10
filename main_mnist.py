from conv_net_mnist import Conv_Net
from tensorflow.keras.datasets import mnist

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

mnist_cnn = Conv_Net(class_names, mnist, 'mnist', color=False, epochs=10)

#mnist_cnn.train()

#mnist_cnn.plot_results(5, 5)

mnist_cnn.save_results_as_pdf()