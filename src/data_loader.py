import numpy as np
import torchvision

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def one_hot_encode(labels, num_classes=10):
    result = np.zeros((len(labels), num_classes), dtype=np.float32)
    result[np.arange(len(labels)), labels] = 1.0
    return result


def preprocess_images(x, normalize_mode="minus1_to_1", to_nchw=True):
    x = x.astype(np.float32)

    if normalize_mode == "zero_to_1":
        x = x / 255.0
    elif normalize_mode == "minus1_to_1":
        x = (x / 255.0 - 0.5) / 0.5
    else:
        raise ValueError("normalize_mode must be 'zero_to_1' or 'minus1_to_1'")

    if to_nchw:
        x = np.transpose(x, (0, 3, 1, 2))

    return x


def load_cifar10(
    root="./data",
    normalize_mode="minus1_to_1",
    to_nchw=True,
    one_hot=True,
    train_samples=None,
    test_samples=None,
):
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

    x_train = trainset.data
    y_train = np.array(trainset.targets)
    x_test = testset.data
    y_test = np.array(testset.targets)

    if train_samples is not None:
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]
    if test_samples is not None:
        x_test = x_test[:test_samples]
        y_test = y_test[:test_samples]

    x_train = preprocess_images(x_train, normalize_mode=normalize_mode, to_nchw=to_nchw)
    x_test = preprocess_images(x_test, normalize_mode=normalize_mode, to_nchw=to_nchw)

    if one_hot:
        y_train = one_hot_encode(y_train, 10)
        y_test = one_hot_encode(y_test, 10)

    return x_train, y_train, x_test, y_test
