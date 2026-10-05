from loguru import logger
import matplotlib.pyplot as plt


def save_figure(fig, path) -> None:
    """
    Salva a figura e libera a memória.

    Args:
        fig (plt.Figure): Figura a ser salva.
        path (Path): Arquivo de destino.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.success(f"Figura salva em: {path}")


def plot_image_grid(images, titles, suptitle, path) -> None:
    """
    Monta uma grade 3x3 de imagens, cada uma com seu título.

    Args:
        images (np.ndarray): Até 9 imagens.
        titles (list[str]): Um título por imagem.
        suptitle (str): Título geral da figura.
        path (Path): Arquivo de destino.
    """
    fig = plt.figure(figsize=(10, 10))

    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(suptitle)

    save_figure(fig, path)


def plot_sample_images(X, y, class_names, path) -> None:
    """
    Mostra as 9 primeiras imagens da base com a classe de cada uma.

    Args:
        X (np.ndarray): Imagens.
        y (np.ndarray): Rótulos.
        class_names (list[str]): Nome de cada classe.
        path (Path): Arquivo de destino.
    """
    plot_image_grid(
        images=X[:9],
        titles=[class_names[label] for label in y[:9]],
        suptitle="Exemplos da base CIFAR-10",
        path=path,
    )


def plot_training_history(history, path) -> None:
    """
    Compara acurácia e loss de treino e validação ao longo das épocas.

    Args:
        history (dict | pd.DataFrame): Histórico do fit, com as chaves
            accuracy, val_accuracy, loss e val_loss.
        path (Path): Arquivo de destino.
    """
    fig, (ax_accuracy, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))

    ax_accuracy.plot(history["accuracy"], label="Treino")
    ax_accuracy.plot(history["val_accuracy"], label="Validação")
    ax_accuracy.set_title("Acurácia por Época")
    ax_accuracy.set_xlabel("Épocas")
    ax_accuracy.set_ylabel("Acurácia")
    ax_accuracy.legend()

    ax_loss.plot(history["loss"], label="Treino")
    ax_loss.plot(history["val_loss"], label="Validação")
    ax_loss.set_title("Loss por Época")
    ax_loss.set_xlabel("Épocas")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    save_figure(fig, path)


def plot_predictions(X, y_true, y_pred, class_names, path) -> None:
    """
    Mostra até 9 imagens com a classe real e a classe predita.

    Args:
        X (np.ndarray): Imagens.
        y_true (np.ndarray): Rótulos reais.
        y_pred (np.ndarray): Rótulos preditos.
        class_names (list[str]): Nome de cada classe.
        path (Path): Arquivo de destino.
    """
    plot_image_grid(
        images=X[:9],
        titles=[
            f"Real: {class_names[real]}\nPredito: {class_names[pred]}"
            for real, pred in zip(y_true[:9], y_pred[:9])
        ],
        suptitle="Previsões do Modelo",
        path=path,
    )
