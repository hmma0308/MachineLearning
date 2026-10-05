from loguru import logger

from module_classificador_imagens.config import (
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from module_classificador_imagens.dataset import (
    CLASS_NAMES,
    load_data,
)
from module_classificador_imagens.features import normalize_images
from module_classificador_imagens.modeling.evaluate import evaluate_model
from module_classificador_imagens.modeling.pipeline import MODEL_NAME
from module_classificador_imagens.modeling.train import train_model
from module_classificador_imagens.plots import (
    plot_sample_images,
    plot_training_history,
)


def main():

    logger.info("Iniciando preparação do dataset...")

    X_train, X_test, y_train, y_test = load_data()

    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)

    plot_sample_images(
        X_train,
        y_train,
        CLASS_NAMES,
        FIGURES_DIR / "sample_images.png",
    )

    # =============================================
    # TREINAMENTO
    # Os últimos 20% do treino viram validação
    # =============================================

    model, history = train_model(
        X_train=X_train,
        y_train=y_train,
        model_path=MODELS_DIR / "cnn_model.keras",
        metadata_path=MODELS_DIR / "metadata.json",
        history_path=REPORTS_DIR / "training_history.csv",
    )

    plot_training_history(
        history.history,
        FIGURES_DIR / "training_history.png",
    )

    # =============================================
    # TESTE FINAL
    # =============================================

    evaluate_model(
        model=model,
        model_name=MODEL_NAME,
        X_test=X_test,
        y_test=y_test,
    )

    logger.success("Pipeline executado com sucesso.")


if __name__ == "__main__":
    main()
