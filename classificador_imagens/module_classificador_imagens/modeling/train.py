import json

from loguru import logger
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from module_classificador_imagens.dataset import CLASS_NAMES
from module_classificador_imagens.modeling.pipeline import (
    INPUT_SHAPE,
    MODEL_NAME,
    create_cnn_model,
)

RANDOM_STATE = 42

EPOCHS = 60
BATCH_SIZE = 64

# Fração do treino usada como validação. O Keras separa os últimos 20%
# das imagens, sem embaralhar, igual ao notebook.
VALIDATION_SPLIT = 0.2


def create_callbacks() -> list:
    """
    Cria os callbacks que controlam o treino.

    Returns:
        list: EarlyStopping e ReduceLROnPlateau.
    """
    # Para o treino quando a val_loss para de melhorar e restaura os melhores pesos.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    # Reduz a taxa de aprendizado pela metade quando a val_loss estagna.
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    return [early_stopping, reduce_lr]


def train_model(
    X_train,
    y_train,
    model_path,
    metadata_path,
    history_path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """
    Treina a CNN e salva modelo, metadados e histórico de treino.

    Args:
        X_train (np.ndarray): Imagens de treino já normalizadas.
        y_train (np.ndarray): Rótulos de treino.
        model_path (Path): Destino do modelo (.keras).
        metadata_path (Path): Destino dos metadados (.json).
        history_path (Path): Destino do histórico por época (.csv).
        epochs (int): Quantidade máxima de épocas.
        batch_size (int): Quantidade de imagens por lote.

    Returns:
        tuple[keras.Sequential, keras.callbacks.History]: Modelo treinado e
            histórico do fit.
    """
    logger.info(f"Treinando modelo: {MODEL_NAME}")

    # Fixa as sementes do Python, do NumPy e do TensorFlow para que o treino
    # seja reproduzível (pesos iniciais, augmentation e dropout).
    keras.utils.set_random_seed(RANDOM_STATE)

    model = create_cnn_model()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=create_callbacks(),
    )

    # -------------------------------------------------
    # Salva o modelo
    # -------------------------------------------------

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.success(f"Modelo salvo em: {model_path}")

    # -------------------------------------------------
    # Salva o histórico de treino
    # -------------------------------------------------

    # Uma linha por época, com loss, accuracy, val_loss, val_accuracy e
    # learning_rate. Permite refazer os gráficos sem treinar de novo.
    history_df = pd.DataFrame(history.history)
    history_df.index.name = "epoch"

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path)
    logger.success(f"Histórico salvo em: {history_path}")

    # -------------------------------------------------
    # Salva metadados
    # -------------------------------------------------

    metadata = {
        "model_name": MODEL_NAME,
        "class_names": CLASS_NAMES,
        "input_shape": list(INPUT_SHAPE),
        "epochs_trained": len(history_df),
        "best_val_loss": float(history_df["val_loss"].min()),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    logger.success(f"Metadados salvos em: {metadata_path}")

    return model, history
