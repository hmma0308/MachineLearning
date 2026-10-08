import pandas as pd
from loguru import logger

from module_Olist.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
)

from module_Olist.modeling.predict import (
    load_model,
    predict,
)


def main():
    logger.info("Carregando dados para inferência...")

    # Carrega o dataset já preparado com o nome correto
    data = pd.read_csv(INTERIM_DATA_DIR / "dataset.csv")

    # Somente as features usadas no treinamento
    X = data[
        [
            "promised_days",
            "item_count",
            "seller_count",
            "total_price",
            "total_freight",
            "purchase_month",
            "purchase_weekday",
            "purchase_hour",
            "customer_state",
        ]
    ]

    # Seleciona algumas amostras
    X_sample = X.sample(n=5, random_state=42)

    # Carrega o artefato do modelo (apenas com o caminho do joblib)
    logger.info("Carregando o artefato do modelo...")
    model, model_name, threshold = load_model(
        model_path=MODELS_DIR / "best_model.joblib"
    )

    # Realiza a inferência
    logger.info("Gerando predições...")
    predictions = predict(
        model=model,
        X=X_sample,
        threshold=threshold,
    )

    logger.info(f"Amostras selecionadas para inferência:\n{X_sample}\n")
    logger.success(f"Predições realizadas:\n{predictions}")


if __name__ == "__main__":
    main()
