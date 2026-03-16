import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.w = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)

    def forward(self, input):
        self.input = input
        return input @ self.w + self.b

    def backward(self, grad_output):
        self.grad_W = self.input.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.w.T

