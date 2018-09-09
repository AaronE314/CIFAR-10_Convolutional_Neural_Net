from conv_net import Conv_Net


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifair = Conv_Net(class_names)

cifair.train()

#cifair.plot_results(5, 5)

cifair.save_results_as_pdf()