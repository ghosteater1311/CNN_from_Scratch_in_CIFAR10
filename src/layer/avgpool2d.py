import numpy as np

class AvgPool2D:
	def __init__(self, kernel_size=2, stride=2):
		self.kernel_size = kernel_size
		self.stride = stride
		self.X_cache = None

	def _to_pair(self, value):
		if isinstance(value, int):
			return value, value
		return value

	def forward(self, X):
		# X: (N, C, H, W)
		N, C, H, W = X.shape
		K_h, K_w = self._to_pair(self.kernel_size)
		S_h, S_w = self._to_pair(self.stride)

		h_num = H - K_h
		w_num = W - K_w
		if h_num < 0 or w_num < 0:
			raise ValueError("Invalid kernel_size: pooling window is larger than input")

		H_out = h_num // S_h + 1
		W_out = w_num // S_w + 1

		out = np.zeros((N, C, H_out, W_out), dtype=X.dtype)

		for n in range(N):
			for c in range(C):
				for i in range(H_out):
					h_start = i * S_h
					h_end = h_start + K_h
					for j in range(W_out):
						w_start = j * S_w
						w_end = w_start + K_w

						window = X[n, c, h_start:h_end, w_start:w_end]
						out[n, c, i, j] = np.mean(window)

		self.X_cache = X
		return out

	def backward(self, dout):
		if self.X_cache is None:
			raise ValueError("forward(X) must be called before backward(dout)")

		X = self.X_cache
		N, C, H, W = X.shape
		K_h, K_w = self._to_pair(self.kernel_size)
		S_h, S_w = self._to_pair(self.stride)

		if dout.shape[:2] != (N, C):
			raise ValueError(f"dout shape mismatch in batch/channel. Expected ({N}, {C}, H_out, W_out), got {dout.shape}")

		H_out, W_out = dout.shape[2], dout.shape[3]
		dX = np.zeros_like(X)
		scale = 1.0 / (K_h * K_w)

		for n in range(N):
			for c in range(C):
				for i in range(H_out):
					h_start = i * S_h
					h_end = h_start + K_h
					for j in range(W_out):
						w_start = j * S_w
						w_end = w_start + K_w

						dX[n, c, h_start:h_end, w_start:w_end] += dout[n, c, i, j] * scale

		return dX
    