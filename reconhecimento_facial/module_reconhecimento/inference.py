import cv2
from loguru import logger

from module_reconhecimento.config import MODELS_DIR
from module_reconhecimento.modeling.predict import (
    load_face_database,
    predict,
)
from module_reconhecimento.plots import draw_help, draw_predictions

# Índice da webcam no OpenCV. 0 é a câmera embutida do notebook.
CAMERA_INDEX = 0

# O reconhecimento roda a cada N frames e, nos demais, o último resultado é
# redesenhado. Gerar embeddings é lento, e isso mantém o vídeo fluido.
RECOGNITION_INTERVAL = 3

WINDOW_NAME = "Controle de Acesso"


def log_access(predictions) -> None:
    """
    Registra no terminal o status de acesso de cada rosto.
    """
    for face in predictions.itertuples():
        if face.authorized:
            logger.success(f"Acesso liberado: {face.name} (distância {face.distance:.2f})")
        else:
            logger.warning(f"Acesso negado (distância {face.distance:.2f})")


def main():

    # Carrega o banco de rostos gerado no cadastro
    database, metadata = load_face_database(
        database_path=MODELS_DIR / "face_database.pkl",
        metadata_path=MODELS_DIR / "metadata.json",
    )

    capture = cv2.VideoCapture(CAMERA_INDEX)

    if not capture.isOpened():
        raise RuntimeError(
            "Não foi possível abrir a webcam. No macOS, libere a câmera para o "
            "terminal/VS Code em Ajustes do Sistema > Privacidade e Segurança > Câmera."
        )

    logger.info("Webcam aberta. Pressione Q na janela do vídeo para sair.")

    frame_count = 0
    last_status = None

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                logger.error("Não foi possível ler o frame da webcam.")
                break

            # Espelha a imagem para o vídeo se comportar como um espelho.
            frame = cv2.flip(frame, 1)

            if frame_count % RECOGNITION_INTERVAL == 0:
                predictions = predict(frame, database, metadata)

                # Só registra no terminal quando o status muda, para não
                # repetir a mesma linha a cada frame.
                status = sorted(zip(predictions["name"], predictions["authorized"]))

                if status != last_status:
                    log_access(predictions)
                    last_status = status

            frame_count += 1

            draw_predictions(frame, predictions)
            draw_help(frame)

            cv2.imshow(WINDOW_NAME, frame)

            # Fecha com Q, Esc ou pelo botão da janela.
            key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1

            if key in (ord("q"), ord("Q"), 27) or window_closed:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    logger.success("Controle de acesso encerrado.")


if __name__ == "__main__":
    main()
