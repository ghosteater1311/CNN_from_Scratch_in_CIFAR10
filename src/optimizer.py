class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def set_lr(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            # Dense-style gradient names
            if hasattr(layer, "w") and hasattr(layer, "b") and hasattr(layer, "grad_W") and hasattr(layer, "grad_b"):
                grad_w = layer.grad_W + self.weight_decay * layer.w
                layer.w -= self.learning_rate * grad_w
                layer.b -= self.learning_rate * layer.grad_b
                continue

            # Conv-style gradient names
            if hasattr(layer, "w") and hasattr(layer, "b") and hasattr(layer, "dW") and hasattr(layer, "db"):
                grad_w = layer.dW + self.weight_decay * layer.w
                layer.w -= self.learning_rate * grad_w
                layer.b -= self.learning_rate * layer.db
