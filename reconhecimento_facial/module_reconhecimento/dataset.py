from pathlib import Path

import cv2
from loguru import logger
import numpy as np
import pandas as pd

# Formatos que o OpenCV consegue ler. Fotos em HEIC (padrão do iPhone)
# precisam ser convertidas para JPG ou PNG antes.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_authorized_images(authorized_dir: Path) -> pd.DataFrame:
    """
    Lista as fotos das pessoas autorizadas.

    Cada pessoa tem uma subpasta com o seu nome, e esse nome é o que aparece
    na webcam quando o acesso é liberado:

        authorized/
        ├── maria/
        │   ├── 1.jpg
        │   └── 2.jpg
        └── joao/
            └── foto.png

    Fotos soltas direto em authorized/ também são aceitas; nesse caso o nome
    da pessoa é o nome do arquivo (joao.jpg -> joao).

    Args:
        authorized_dir (Path): Pasta com as fotos das pessoas autorizadas.

    Returns:
        pd.DataFrame: Uma linha por foto, com as colunas:
            - name: nome da pessoa.
            - image_path: caminho da foto.
    """
    logger.info(f"Carregando fotos de: {authorized_dir}")

    rows = []

    for path in sorted(authorized_dir.rglob("*")):
        # Ignora pastas e arquivos ocultos (.DS_Store, .gitkeep).
        if path.is_dir() or path.name.startswith("."):
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            logger.warning(f"Formato não suportado, arquivo ignorado: {path.name}")
            continue

        # A primeira pasta abaixo de authorized/ identifica a pessoa.
        parts = path.relative_to(authorized_dir).parts
        name = parts[0] if len(parts) > 1 else path.stem

        rows.append({"name": name, "image_path": path})

    images = pd.DataFrame(rows, columns=["name", "image_path"])

    if images.empty:
        raise FileNotFoundError(
            f"Nenhuma foto encontrada em {authorized_dir}. "
            "Crie uma subpasta por pessoa com as fotos dela (ex.: authorized/maria/1.jpg)."
        )

    logger.info(f"Fotos encontradas: {len(images)}")
    logger.info(f"Fotos por pessoa:\n{images['name'].value_counts()}")

    return images


def read_image(image_path: Path) -> np.ndarray:
    """
    Lê uma imagem do disco no formato BGR, o mesmo da webcam no OpenCV.

    O DeepFace recusa caminhos com acentos (ex.: authorized/joão/1.jpg), e o
    cv2.imread também falha com eles no Windows. Ler os bytes com o numpy e
    decodificar com o OpenCV funciona com qualquer caminho.

    Args:
        image_path (Path): Caminho da imagem.

    Returns:
        np.ndarray: Imagem no formato (altura, largura, 3), em BGR.
    """
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")

    return image
