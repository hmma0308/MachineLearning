from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    RandomFlip,
    RandomTranslation,
)

from module_classificador_imagens.dataset import CLASS_NAMES

MODEL_NAME = "CNN"

# Cada imagem do CIFAR-10 tem 32x32 pixels e 3 canais de cor (RGB).
INPUT_SHAPE = (32, 32, 3)

NUM_CLASSES = len(CLASS_NAMES)


def create_augmentation() -> keras.Sequential:
    """
    Cria as camadas de data augmentation.

    Essas camadas só alteram as imagens durante o treino. Na avaliação e na
    inferência, a imagem passa por elas sem mudança.

    Returns:
        keras.Sequential: Bloco de data augmentation.
    """
    return keras.Sequential(
        [
            # Espelhamento horizontal aleatório.
            RandomFlip("horizontal"),
            # Deslocamento aleatório de até 10% na altura e na largura.
            RandomTranslation(height_factor=0.1, width_factor=0.1),
        ],
        name="data_augmentation",
    )


def add_conv_block(model: keras.Sequential, filters: int, dropout: float) -> None:
    """
    Adiciona um bloco convolucional ao modelo.

    Args:
        model (keras.Sequential): Modelo que recebe o bloco.
        filters (int): Quantidade de filtros das duas convoluções.
        dropout (float): Fração de neurônios desligados ao final do bloco.
    """
    # padding='same' mantém o tamanho da imagem após a convolução.
    model.add(Conv2D(filters=filters, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=filters, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())

    # Reduz a imagem pela metade (32 -> 16 -> 8 -> 4 ao longo dos blocos).
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))


def create_cnn_model() -> keras.Sequential:
    """
    Cria e compila a CNN usada para classificar as imagens.

    Returns:
        keras.Sequential: Modelo compilado, pronto para o fit.
    """
    model = keras.Sequential(name="cnn")

    # Camada de entrada
    model.add(Input(shape=INPUT_SHAPE))

    # Data augmentation
    model.add(create_augmentation())

    # Três blocos convolucionais: os filtros dobram e o dropout aumenta
    # a cada bloco para conter o overfitting.
    add_conv_block(model, filters=32, dropout=0.2)
    add_conv_block(model, filters=64, dropout=0.3)
    add_conv_block(model, filters=128, dropout=0.4)

    # Transição CNN -> Dense (totalmente conectada)
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    # Uma saída por classe; o softmax transforma em probabilidades.
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
