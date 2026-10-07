import unicodedata

import cv2

# Cores no formato BGR, a ordem de canais usada pelo OpenCV (não RGB).
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2


def to_ascii(text: str) -> str:
    """
    Remove os acentos do texto (João -> Joao).

    As fontes do cv2.putText só desenham caracteres ASCII; letras acentuadas
    apareceriam como "?" na tela.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def draw_access_box(frame, x, y, w, h, label, color) -> None:
    """
    Desenha o bounding box de um rosto com o status de acesso acima dele.

    Args:
        frame (np.ndarray): Frame da webcam (BGR), alterado no lugar.
        x, y, w, h (int): Posição e tamanho do rosto no frame.
        label (str): Texto exibido acima do box.
        color (tuple[int, int, int]): Cor do box em BGR.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, THICKNESS)

    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
    label_w = text_w + 8
    label_h = text_h + baseline + 8

    # Faixa na cor do box atrás do texto, para facilitar a leitura. Se o rosto
    # estiver colado no topo ou na direita da imagem, a faixa é deslocada para
    # não ficar cortada.
    label_top = max(y - label_h, 0)
    label_left = max(min(x, frame.shape[1] - label_w), 0)

    cv2.rectangle(
        frame,
        (label_left, label_top),
        (label_left + label_w, label_top + label_h),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (label_left + 4, label_top + text_h + 4),
        FONT,
        FONT_SCALE,
        BLACK,
        THICKNESS,
    )


def draw_predictions(frame, predictions) -> None:
    """
    Desenha um bounding box por rosto: verde para acesso liberado
    e vermelho para acesso negado.

    Args:
        frame (np.ndarray): Frame da webcam (BGR), alterado no lugar.
        predictions (pd.DataFrame): Saída do predict, com name, authorized, x, y, w e h.
    """
    for face in predictions.itertuples():
        if face.authorized:
            label = f"ACESSO LIBERADO: {to_ascii(face.name)}"
            color = GREEN
        else:
            label = "ACESSO NEGADO"
            color = RED

        draw_access_box(
            frame,
            int(face.x),
            int(face.y),
            int(face.w),
            int(face.h),
            label,
            color,
        )


def draw_help(frame) -> None:
    """
    Escreve no canto inferior do frame como fechar a janela.

    Args:
        frame (np.ndarray): Frame da webcam (BGR), alterado no lugar.
    """
    cv2.putText(
        frame,
        "Pressione Q para sair",
        (10, frame.shape[0] - 10),
        FONT,
        0.5,
        WHITE,
        1,
    )
