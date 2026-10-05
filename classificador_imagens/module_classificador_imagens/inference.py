from loguru import logger
import numpy as np

from module_classificador_imagens.config import (
    FIGURES_DIR,
    MODELS_DIR,
)
from module_classificador_imagens.dataset import load_data
from module_classificador_imagens.features import normalize_images
from module_classificador_imagens.modeling.predict import (
    load_model,
    predict,
)
from module_classificador_imagens.plots import plot_predictions

# Quantidade de imagens do teste usadas na inferência (grade 3x3).
N_SAMPLES = 9


def main():

    # Carrega somente o conjunto de teste
    _, X_test, _, y_test = load_data()

    # Seleciona algumas amostras
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(X_test), size=N_SAMPLES, replace=False)

    # Mesma normalização aplicada no treinamento
    X_sample = normalize_images(X_test[sample_indices])
    y_sample = y_test[sample_indices]

    # Carrega o modelo já treinado
    model, _, class_names = load_model(
        model_path=MODELS_DIR / "cnn_model.keras",
        metadata_path=MODELS_DIR / "metadata.json",
    )

    # Realiza a inferência
    predictions = predict(
        model=model,
        X=X_sample,
        class_names=class_names,
    )

    # Posição de cada imagem no teste e sua classe real,
    # para comparar com a predição
    predictions.index = sample_indices
    predictions.index.name = "test_index"
    predictions.insert(0, "real", [class_names[label] for label in y_sample])

    logger.success(f"Predições realizadas:\n{predictions}")

    plot_predictions(
        X_sample,
        y_sample,
        predictions["prediction"].to_numpy(),
        class_names,
        FIGURES_DIR / "predictions.png",
    )


if __name__ == "__main__":
    main()
