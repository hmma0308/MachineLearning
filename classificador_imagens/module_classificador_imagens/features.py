import numpy as np


def normalize_images(X: np.ndarray) -> np.ndarray:
    """
    Normaliza os pixels das imagens para o intervalo [0, 1].

    Args:
        X (np.ndarray): Imagens com pixels de 0 a 255 (uint8).

    Returns:
        np.ndarray: Imagens em float32 com pixels de 0 a 1.
    """
    # Cada pixel vai de 0 a 255. Dividir por 255 coloca todos os valores
    # entre 0 e 1, escala em que a rede treina de forma mais estável.
    return X.astype("float32") / 255.0
