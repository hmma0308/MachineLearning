import json

from deepface import DeepFace
from deepface.modules.verification import find_distance
from loguru import logger
import numpy as np
import pandas as pd

from module_reconhecimento.features import extract_faces

# Nome exibido para rostos que não batem com ninguém do banco.
UNKNOWN_NAME = "Desconhecido"


def load_face_database(database_path, metadata_path):
    """
    Carrega o banco de rostos autorizados
    e os metadados da inferência.
    """
    if not database_path.exists():
        raise FileNotFoundError(
            f"Banco de rostos não encontrado em {database_path}. "
            "Rode o cadastro antes com: make train"
        )

    database = pd.read_pickle(database_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Carrega os pesos agora para a webcam não travar no primeiro frame.
    DeepFace.build_model(model_name=metadata["model_name"])

    logger.info(f"Modelo carregado: {metadata['model_name']}")
    logger.info(f"Threshold carregado: {metadata['threshold']:.2f}")
    logger.info(f"Pessoas autorizadas: {sorted(database['name'].unique())}")

    return (database, metadata)


def identify_faces(embeddings, database, distance_metric, threshold):
    """
    Compara cada rosto com o banco e decide se o acesso é liberado.

    A foto do banco mais parecida (menor distância) define quem é a pessoa,
    e o acesso só é liberado se essa distância não passar do threshold.

    Args:
        embeddings (np.ndarray): Embeddings dos rostos, formato (rostos, dimensões).
        database (pd.DataFrame): Banco com as colunas name e embedding.
        distance_metric (str): Métrica usada no cadastro.
        threshold (float): Distância máxima para liberar o acesso.

    Returns:
        pd.DataFrame: Uma linha por rosto, com as colunas name, distance e authorized.
    """
    # Distância de cada rosto (linhas) para cada foto do banco (colunas).
    distances = find_distance(
        np.stack(database["embedding"].to_numpy()),
        embeddings,
        distance_metric,
    )

    # A foto mais parecida com cada rosto é a de menor distância.
    closest = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)
    authorized = min_distances <= threshold

    return pd.DataFrame(
        {
            "name": np.where(authorized, database["name"].to_numpy()[closest], UNKNOWN_NAME),
            "distance": min_distances,
            "authorized": authorized,
        }
    )


def predict(image, database, metadata):
    """
    Reconhece os rostos de uma imagem e decide o acesso de cada um.

    Returns:
        pd.DataFrame: Uma linha por rosto, com name, distance, authorized e a
            posição do rosto (x, y, w, h), usada para desenhar o bounding box.
    """
    faces = extract_faces(
        image,
        model_name=metadata["model_name"],
        detector_backend=metadata["detector_backend"],
    )

    if not faces:
        return pd.DataFrame(columns=["name", "distance", "authorized", "x", "y", "w", "h"])

    predictions = identify_faces(
        embeddings=np.stack([face["embedding"] for face in faces]),
        database=database,
        distance_metric=metadata["distance_metric"],
        threshold=metadata["threshold"],
    )

    facial_areas = pd.DataFrame([face["facial_area"] for face in faces])

    return pd.concat([predictions, facial_areas[["x", "y", "w", "h"]]], axis=1)
