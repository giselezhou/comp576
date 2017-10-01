from three_layer_neural_network import NeuralNetwork, plot_decision_boundary
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import OneHotEncoder


def generate_data(type=None):
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    if type == 'make_moons':
        X, y = datasets.make_moons(200, noise=0.20)
    else:
        X, y = datasets.make_circles(200, noise=0.20)
    return X, y



class Layer(object):
    def __init__(self,  l_input_dim, l_output_dim, actFun_type):
        self.l_input_dim = l_input_dim
        self.l_output_dim = l_output_dim
        self.W = np.random.randn(self.l_input_dim, self.l_output_dim) / np.sqrt(self.l_input_dim)
        self.b = np.zeros((1, self.l_output_dim))
        self.actFun_type = actFun_type
        self.a = None
        self.z = None
        self.X = None
        self.delta = None
        self.db = None
        self.dW = None

    def feedforward(self, X=None, z=None):
        if z is None:
            self.a = X
            return self.a.dot(self.W) + self.b
        else:
            self.z = z
            self.a = NeuralNetwork.actFun(self.z, self.actFun_type)
            return self.a.dot(self.W) + self.b

    def backprop(self, post_delta, num_examples):
        self.dW = 1. / num_examples * self.a.T.dot(post_delta)
        self.db = 1. / num_examples * np.sum(post_delta, axis=0)
        if self.z is not None:

            self.delta = post_delta.dot(self.W.T) * NeuralNetwork.diff_actFun(self.z, self.actFun_type)
        return self.delta


class DeepNeuralNetwork(object):

    def __init__(self, nn_layer_sizes, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_num_layer: number of layers of neural network (input and output inclusive)
        :param nn_layer_sizes: dimension of each layer from input to hidden, to output
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_layer_sizes = nn_layer_sizes
        self.nn_num_layer = len(nn_layer_sizes)
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.layers = []
        np.random.seed(seed)

        # initialize the layers in the network
        for i in range(self.nn_num_layer - 1):
            layer = Layer(nn_layer_sizes[i], nn_layer_sizes[i+1], self.actFun_type)
            self.layers.append(layer)

    def feedforward(self, X):
        a = X
        z = self.layers[0].feedforward(X=a)
        for i in range(1, len(self.layers), 1):
            z = self.layers[i].feedforward(z=z)
        exp_scores = np.exp(z)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        out = self.feedforward(X)
        _, num_features = out.shape
        # Calculating the loss
        v = OneHotEncoder(n_values=num_features, sparse=False).fit_transform(y.reshape(-1, 1))
        data_loss = -1 * np.sum(np.log(out) * v)

        # Add regulationzation term to loss (optional)
        m_sum = 0
        for layer in self.layers:
            m_sum += np.sum(np.square(layer.W))
        data_loss += self.reg_lambda / 2 * m_sum
        return (1. / num_examples) * data_loss

    def backprop(self, X, y):
        num = len(X)
        last_delta = self.probs
        last_delta[range(num), y] -= 1.0
        for i in range(len(self.layers) - 1, -1, -1):
            last_delta = self.layers[i].backprop(last_delta, num)

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            self.backprop(X, y)
            # Add regularization terms (b1 and b2 don't have regularization terms)
            for layer in self.layers:
                layer.dW += self.reg_lambda * layer.W

            # Gradient descent parameter update
            for layer in self.layers:
                layer.W += -epsilon * layer.dW
                layer.b += -epsilon * layer.db

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    # # generate and visualize Make-Moons dataset
    X, y = generate_data()
    num_samples, dim = X.shape

    model = DeepNeuralNetwork(nn_layer_sizes=[dim, 10, 5, 10, dim], actFun_type='tanh')

    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()