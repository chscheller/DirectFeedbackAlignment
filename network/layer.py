
class Layer:

    def initialize(self, input_size, out_layer_size, train_method):
        """
        Initializes all hyper parameters of this layer
        :param input_size: 
        :type input_size: tuple
        :param out_layer_size: 
        :type out_layer_size: int
        :param train_method: 
        :type train_method: str
        :return: The output dimensions of this layer
        :rtype: tuple
        """
        pass

    def forward(self, X):
        pass

    def dfa(self, e, reg, lr):
        return self

    def back_prob(self, e, reg, lr):
        pass

    def sum_weights(self):
        return 0
