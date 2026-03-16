import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if isinstance(self.kernel_size, int):
            K_h, K_w = self.kernel_size, self.kernel_size
        else:
            K_h, K_w = self.kernel_size

        self.w = np.random.randn(self.out_channels, self.in_channels, K_h, K_w) * 0.01
        self.b = np.zeros((self.out_channels,), dtype=np.float32)
        self.dW = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.X_cache = None

    def forward(self, X):
        N, C, H, W = X.shape
        self.X_cache = X

        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, but got C={C}")

        # Support int or tuple for kernel_size
        if isinstance(self.kernel_size, int):
            K_h, K_w = self.kernel_size, self.kernel_size
        else:
            K_h, K_w = self.kernel_size

        # Support int or tuple for stride
        if isinstance(self.stride, int):
            S_h, S_w = self.stride, self.stride
        else:
            S_h, S_w = self.stride

        # Support int or tuple for padding
        if isinstance(self.padding, int):
            P_h, P_w = self.padding, self.padding
        else:
            P_h, P_w = self.padding

        # Zero-padding 
        if P_h > 0 or P_w > 0:
            X_padded = np.pad(
                X,
                pad_width=((0, 0), (0, 0), (P_h, P_h), (P_w, P_w)),
                mode="constant",
                constant_values=0
            )
        else:
            X_padded = X

        h_num = H + 2 * P_h - K_h
        w_num = W + 2 * P_w - K_w

        if h_num < 0 or w_num < 0:
            raise ValueError("Invalid kernel_size/padding: output spatial size becomes negative")

        H_out = h_num // S_h + 1
        W_out = w_num // S_w + 1

        out = np.zeros((N, self.out_channels, H_out, W_out), dtype=X.dtype)

        # Convolution loops
        for n in range(N):
            for f in range(self.out_channels):
                for i in range(H_out):
                    h_start = i * S_h
                    h_end = h_start + K_h
                    for j in range(W_out):
                        w_start = j * S_w
                        w_end = w_start + K_w

                        patch = X_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, f, i, j] = np.sum(patch * self.w[f]) + self.b[f]

        return out

    def backward(self, dout):
        if self.X_cache is None:
            raise ValueError("forward(X) must be called before backward(dout)")

        X = self.X_cache
        N, C, H, W = X.shape

        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, but got C={C}")

        # Support int or tuple for kernel_size
        if isinstance(self.kernel_size, int):
            K_h, K_w = self.kernel_size, self.kernel_size
        else:
            K_h, K_w = self.kernel_size

        # Support int or tuple for stride
        if isinstance(self.stride, int):
            S_h, S_w = self.stride, self.stride
        else:
            S_h, S_w = self.stride

        # Support int or tuple for padding
        if isinstance(self.padding, int):
            P_h, P_w = self.padding, self.padding
        else:
            P_h, P_w = self.padding

        # Build padded input
        if P_h > 0 or P_w > 0:
            X_padded = np.pad(
                X,
                pad_width=((0, 0), (0, 0), (P_h, P_h), (P_w, P_w)),
                mode="constant",
                constant_values=0
            )
        else:
            X_padded = X

        # Validate dout shape
        expected_h_out = (H + 2 * P_h - K_h) // S_h + 1
        expected_w_out = (W + 2 * P_w - K_w) // S_w + 1
        if dout.shape != (N, self.out_channels, expected_h_out, expected_w_out):
            raise ValueError(f"dout shape mismatch. Expected {(N, self.out_channels, expected_h_out, expected_w_out)}, got {dout.shape}")

        # Initialize gradients
        self.dW = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        dX_padded = np.zeros_like(X_padded)

        H_out, W_out = expected_h_out, expected_w_out

        for n in range(N):
            for f in range(self.out_channels):
                for i in range(H_out):
                    h_start = i * S_h
                    h_end = h_start + K_h
                    for j in range(W_out):
                        w_start = j * S_w
                        w_end = w_start + K_w

                        dout_val = dout[n, f, i, j]
                        patch = X_padded[n, :, h_start:h_end, w_start:w_end]

                        self.db[f] += dout_val
                        self.dW[f] += patch * dout_val
                        dX_padded[n, :, h_start:h_end, w_start:w_end] += self.w[f] * dout_val

        # Remove padding from dX
        if P_h > 0 or P_w > 0:
            dX = dX_padded[:, :, P_h:P_h + H, P_w:P_w + W]
        else:
            dX = dX_padded

        return dX


