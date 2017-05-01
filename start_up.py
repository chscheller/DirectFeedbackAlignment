from multiprocessing import freeze_support

from models.cifar10.simple_conv_3_bp import simple_conv_3_bp

if __name__ == '__main__':
    freeze_support()

    model = simple_conv_3_bp()
    # model = simple_conv_3_dfa()

    # model = mnist_simple_conv_3_bp()