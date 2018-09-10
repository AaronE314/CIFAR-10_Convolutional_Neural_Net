from conv_net_cifar10 import Conv_Net
from tensorflow.keras.datasets import cifar10


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifair_cnn = Conv_Net(class_names, cifar10, 'cifar10')

#cifair.train()

#cifair.plot_results(5, 5)


cifair_cnn.save_results_as_pdf()