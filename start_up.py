from multiprocessing import freeze_support

from models.cifar10.simple_conv_100_bp import simple_conv_100_bp
from models.cifar10.simple_conv_100_dfa import simple_conv_100_dfa
from models.cifar10.simple_conv_3_bp import simple_conv_3_bp
from models.cifar10.simple_conv_3_dfa import simple_conv_3_dfa
from models.cifar10.simple_conv_3_do_bp import simple_conv_3_do_bp

if __name__ == '__main__':
    freeze_support()

    """ simple_conv_3 """
    model = simple_conv_3_bp()
    # model = simple_conv_3_dfa()

    """ simple_conv_3_do """
    model = simple_conv_3_do_bp()

    """ simple_conv_100 """
    # model = simple_conv_100_bp()
    # model = simple_conv_100_dfa()

    # model = mnist_simple_conv_3_bp()