from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
import numpy as np


def extract_faces(image: np.ndarray, model_name: str, detector_backend: str) -> list[dict]:
    """
    Detecta os rostos da imagem e gera o embedding de cada um.

    O embedding é um vetor numérico que resume as características do rosto.
    Fotos da mesma pessoa geram vetores próximos, e de pessoas diferentes,
    vetores distantes. É a feature usada para comparar rostos.

    Args:
        image (np.ndarray): Imagem em BGR (foto lida do disco ou frame da webcam).
        model_name (str): Modelo do DeepFace que gera o embedding.
        detector_backend (str): Detector do DeepFace que localiza os rostos.

    Returns:
        list[dict]: Um item por rosto encontrado, com as chaves:
            - embedding: vetor do rosto (np.ndarray).
            - facial_area: posição do rosto na imagem (x, y, w, h).
            Lista vazia quando não há rosto na imagem.
    """
    try:
        faces = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            # Sem rosto, o DeepFace lança FaceNotDetected. Com False, ele
            # trataria a imagem inteira como um rosto.
            enforce_detection=True,
        )
    except FaceNotDetected:
        return []

    return [
        {
            "embedding": np.asarray(face["embedding"], dtype=np.float32),
            "facial_area": face["facial_area"],
        }
        for face in faces
    ]
