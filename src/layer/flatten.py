import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        # X: (N, C, H, W)
        self.input_shape = X.shape
        N = X.shape[0]
        return X.reshape(N, -1)

    def backward(self, dout):
        if self.input_shape is None:
            raise ValueError("forward(X) must be called before backward(dout)")

        expected_features = int(np.prod(self.input_shape[1:]))
        if dout.ndim != 2 or dout.shape[0] != self.input_shape[0] or dout.shape[1] != expected_features:
            raise ValueError(f"dout shape mismatch. Expected ({self.input_shape[0]}, {expected_features}), got {dout.shape}")

        return dout.reshape(self.input_shape)
