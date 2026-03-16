import os
import numpy as np

from data_loader import load_cifar10
from layer.conv2d import Conv2D
from layer.maxpool2d import MaxPool2D
from layer.flatten import Flatten
from layer.dense import DenseLayer
from layer.activations import relu, softmax
from loss import cross_entropy_loss
from optimizer import SGD


def forward_pass(x_batch, conv, pool, flatten, dense):
    conv_out = conv.forward(x_batch)
    relu_out = relu(conv_out)
    pool_out = pool.forward(relu_out)
    flat_out = flatten.forward(pool_out)
    logits = dense.forward(flat_out)
    probs = softmax(logits)
    return conv_out, probs


def evaluate_accuracy(x_eval, y_eval, conv, pool, flatten, dense):
    _, probs = forward_pass(x_eval, conv, pool, flatten, dense)
    return np.mean(np.argmax(probs, axis=1) == np.argmax(y_eval, axis=1))


def horizontal_flip_batch(x_batch, p=0.5):
    # x_batch: NCHW
    mask = np.random.rand(x_batch.shape[0]) < p
    if np.any(mask):
        x_batch[mask] = x_batch[mask, :, :, ::-1]
    return x_batch


def main():
    # Profiles
    QUICK_RUN = False

    if QUICK_RUN:
        learning_rate = 0.01
        epochs = 1
        batch_size = 64
        num_train_samples = 1000
        num_test_samples = 200
        eval_samples = 200
    else:
        learning_rate = 0.01
        epochs = 5
        batch_size = 32
        num_train_samples = 10000
        num_test_samples = 2000
        eval_samples = 1000

    # LR schedule (Phase 4.3): step decay
    lr_decay_every = 5
    lr_decay_factor = 0.5

    # Regularization (Phase 4.4): light L2 weight decay
    weight_decay = 1e-4

    x_train, y_train, x_test, y_test = load_cifar10(
        root="./data",
        normalize_mode="minus1_to_1",
        to_nchw=True,
        one_hot=True,
        train_samples=num_train_samples,
        test_samples=num_test_samples,
    )

    # Model: Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax
    conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
    pool = MaxPool2D(kernel_size=2, stride=2)
    flatten = Flatten()
    dense = DenseLayer(input_size=8 * 16 * 16, output_size=10)
    optimizer = SGD(learning_rate=learning_rate, weight_decay=weight_decay)

    history = {"loss": [], "train_acc": [], "test_acc": [], "lr": []}
    num_batches = len(x_train) // batch_size

    for epoch in range(epochs):
        if epoch > 0 and epoch % lr_decay_every == 0:
            optimizer.set_lr(optimizer.learning_rate * lr_decay_factor)

        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0.0

        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            x_batch = x_train[start:end].copy()
            y_batch = y_train[start:end]

            # Light augmentation
            x_batch = horizontal_flip_batch(x_batch, p=0.5)

            # Forward
            conv_out, predictions = forward_pass(x_batch, conv, pool, flatten, dense)

            # Loss
            loss = cross_entropy_loss(predictions, y_batch)
            epoch_loss += loss

            # Backward
            grad = (predictions - y_batch) / batch_size
            grad = dense.backward(grad)
            grad = flatten.backward(grad)
            grad = pool.backward(grad)
            grad = grad * (conv_out > 0)  # ReLU backward mask
            _ = conv.backward(grad)

            # SGD update per batch
            optimizer.step([conv, dense])

            if b % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {b}/{num_batches} | "
                    f"Loss: {loss:.4f} | LR: {optimizer.learning_rate:.5f}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)

        train_acc = evaluate_accuracy(
            x_train[:eval_samples], y_train[:eval_samples], conv, pool, flatten, dense
        )
        test_acc = evaluate_accuracy(
            x_test[:eval_samples], y_test[:eval_samples], conv, pool, flatten, dense
        )

        history["loss"].append(float(avg_loss))
        history["train_acc"].append(float(train_acc))
        history["test_acc"].append(float(test_acc))
        history["lr"].append(float(optimizer.learning_rate))

        print(
            f"Epoch {epoch + 1} done | Avg Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | Test Acc: {test_acc * 100:.2f}%"
        )

if __name__ == "__main__":
    main()
