# module_Olist/modeling/train.py
import joblib
import pandas as pd
from loguru import logger

from module_Olist.config import MODELS_DIR
from module_Olist.cross_validation import prepare_data
from module_Olist.cross_validation_2 import cross_validate_models

# Importa os pipelines da sua pasta modeling
from module_Olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)

def get_pipeline_by_name(name: str):
    """Retorna a instância do pipeline com base no nome do modelo."""
    pipelines = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }
    return pipelines[name]

def train_and_save_best_model(df: pd.DataFrame) -> None:
    """
    Prepara os dados, executa a validação cruzada para encontrar o melhor
    modelo e limiar (threshold), treina o vencedor na base completa e salva.
    """
    logger.info("Preparando os dados para seleção e treinamento...")
    X, y = prepare_data(df)

    logger.info("Iniciando a busca pelo melhor modelo (Cross-Validation)...")
    # Usa a função do cross_validation_2.py para encontrar o vencedor
    best_model_name, best_threshold = cross_validate_models(X, y)
    
    logger.info(f"Treinando o modelo vencedor ({best_model_name}) em todos os dados (Full Train)...")
    best_pipeline = get_pipeline_by_name(best_model_name)
    best_pipeline.fit(X, y)

    # Empacota o modelo, o threshold ideal e o nome das colunas
    model_artifact = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "optimal_threshold": best_threshold,
        "features": list(X.columns)
    }

    # Garante que a pasta models existe e salva o arquivo
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.joblib"
    
    logger.info(f"Salvando o artefato do modelo em {model_path}...")
    joblib.dump(model_artifact, model_path)
    
    logger.success("Treinamento finalizado e modelo salvo com sucesso!")