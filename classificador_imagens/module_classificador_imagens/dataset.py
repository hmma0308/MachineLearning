from loguru import logger
import numpy as np
from tensorflow import keras

# Nomes das classes na mesma ordem dos rótulos do CIFAR-10:
# o rótulo 0 é "avião", o 1 é "automóvel", e assim por diante.
CLASS_NAMES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega a base CIFAR-10 pelo Keras.

    Na primeira execução o Keras baixa a base (~170 MB) e guarda em
    ~/.keras/datasets. Nas execuções seguintes, lê direto do cache.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Uma tupla contendo:
            - X_train: imagens de treino, formato (50000, 32, 32, 3), uint8.
            - X_test: imagens de teste, formato (10000, 32, 32, 3), uint8.
            - y_train: rótulos de treino, formato (50000,).
            - y_test: rótulos de teste, formato (10000,).
    """
    logger.info("Carregando a base CIFAR-10...")

    # A base já vem separada em treino e teste.
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # O Keras entrega os rótulos como coluna, no formato (n, 1).
    # O flatten transforma em vetor (n,), o que permite usar
    # CLASS_NAMES[y[i]] e comparar direto com as predições.
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    logger.info(f"Formato de X_train: {X_train.shape}")
    logger.info(f"Formato de y_train: {y_train.shape}")
    logger.info(f"Formato de X_test: {X_test.shape}")
    logger.info(f"Formato de y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test
