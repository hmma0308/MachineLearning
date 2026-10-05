import json

from loguru import logger
import numpy as np
import pandas as pd
from tensorflow import keras


def load_model(model_path, metadata_path):
    """
    Carrega o modelo treinado
    e os metadados da inferência.
    """
    model = keras.models.load_model(model_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_name = metadata["model_name"]
    class_names = metadata["class_names"]

    logger.info(f"Modelo carregado: {model_name}")
    logger.info(f"Classes carregadas: {class_names}")

    return (model, model_name, class_names)


def predict(model, X, class_names):
    """
    Realiza inferência e retorna, para cada imagem,
    a classe com maior probabilidade.
    """
    # Probabilidade de cada classe para cada imagem: formato (n, 10).
    y_proba = model.predict(X, verbose=0)

    # A classe predita é a de maior probabilidade.
    y_pred = np.argmax(y_proba, axis=1)

    predictions = pd.DataFrame(
        {
            "prediction": y_pred,
            "class_name": [class_names[i] for i in y_pred],
            "confidence": y_proba.max(axis=1),
        }
    )

    return predictions
