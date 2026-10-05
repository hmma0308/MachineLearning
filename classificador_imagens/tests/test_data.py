import numpy as np
import pytest

from module_classificador_imagens.dataset import CLASS_NAMES, load_data
from module_classificador_imagens.features import normalize_images

N_TRAIN = 50_000
N_TEST = 10_000
IMAGE_SHAPE = (32, 32, 3)


@pytest.fixture(scope="module")
def cifar10():
    # Carrega a base uma única vez para todos os testes do arquivo.
    return load_data()


def test_images_have_expected_shape_and_dtype(cifar10):
    X_train, X_test, _, _ = cifar10

    assert X_train.shape == (N_TRAIN, *IMAGE_SHAPE)
    assert X_test.shape == (N_TEST, *IMAGE_SHAPE)
    assert X_train.dtype == np.uint8
    assert X_test.dtype == np.uint8


def test_labels_are_flat_and_valid(cifar10):
    _, _, y_train, y_test = cifar10

    assert y_train.shape == (N_TRAIN,)
    assert y_test.shape == (N_TEST,)

    for y in (y_train, y_test):
        assert y.min() >= 0
        assert y.max() < len(CLASS_NAMES)


def test_train_classes_are_balanced(cifar10):
    _, _, y_train, _ = cifar10

    counts = np.bincount(y_train, minlength=len(CLASS_NAMES))

    assert (counts == N_TRAIN // len(CLASS_NAMES)).all()


def test_normalized_images_are_between_zero_and_one(cifar10):
    _, X_test, _, _ = cifar10

    X_normalized = normalize_images(X_test)

    assert X_normalized.dtype == np.float32
    assert X_normalized.min() >= 0.0
    assert X_normalized.max() <= 1.0
