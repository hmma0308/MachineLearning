from loguru import logger


def evaluate_model(
    model,
    model_name,
    X_test,
    y_test,
):
    """
    Realiza a avaliação final do modelo
    no conjunto de teste.
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # Resultados Finais
    logger.info("=" * 60)
    logger.success(f"MODELO FINAL: {model_name}")
    logger.info(f"Loss: {test_loss:.3f}")
    logger.info(f"Accuracy: {test_accuracy:.3f}")

    return {
        "model_name": model_name,
        "loss": test_loss,
        "accuracy": test_accuracy,
    }
