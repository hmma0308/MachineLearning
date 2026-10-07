from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Fotos das pessoas autorizadas: uma subpasta por pessoa, com o nome dela.
AUTHORIZED_DIR = RAW_DATA_DIR / "authorized"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reconhecimento facial (DeepFace)
# Modelo que transforma cada rosto em um embedding de 512 posições.
MODEL_NAME = "Facenet512"

# Detector de rostos. O YuNet é rápido o bastante para a webcam em tempo real
# e encontra mais rostos que o "opencv" (Haar cascade).
DETECTOR_BACKEND = "yunet"

# Métrica usada para comparar dois embeddings.
DISTANCE_METRIC = "cosine"

# Distância máxima para considerar dois rostos a mesma pessoa. None usa o valor
# calibrado pelo DeepFace para o par modelo + métrica (0.30 no Facenet512 com cosseno).
# Aumente se pessoas autorizadas forem negadas; diminua se desconhecidos forem liberados.
THRESHOLD = None

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
