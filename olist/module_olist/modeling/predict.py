# module_Olist/modeling/predict.py
import joblib
import pandas as pd
from loguru import logger


def load_model(model_path):
    """
    Carrega o artefato do modelo treinado, que já contém
    o pipeline, o threshold ideal e o nome do modelo.
    """
    logger.info(f"Carregando artefato do modelo em: {model_path}")

    # Carrega o dicionário único
    artifact = joblib.load(model_path)

    model = artifact["pipeline"]
    model_name = artifact["model_name"]
    threshold = float(artifact["optimal_threshold"])

    logger.info(f"Modelo carregado: {model_name}")
    logger.info(f"Threshold carregado: {threshold:.2f}")

    return (model, model_name, threshold)


def predict(model, X, threshold):
    """
    Realiza inferência utilizando o threshold definido na validação.
    """
    # Pega as probabilidades apenas da classe 1 (atraso)
    y_proba = model.predict_proba(X)[:, 1]

    # Aplica o threshold otimizado
    y_pred = (y_proba >= threshold).astype(int)

    predictions = pd.DataFrame(
        {
            "prob_is_late": y_proba,
            "prediction": y_pred,
        },
        index=X.index,
    )

    return predictions
