from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

"""
# Comparação

| Aspecto | **GradientBoosting** (sklearn) | **XGBoost** | **LightGBM** |
| --- | --- | --- | --- |
| **Crescimento da árvore** | Level-wise | Level-wise | **Leaf-wise** (mais rápido) |
| **Velocidade** | Lento | Rápido | **Mais rápido** |
| **Uso de memória** | Alto | Médio | **Baixo** |
| **Datasets grandes** | Não escala bem | Escala bem | **Escala melhor** |
| **Regularização** | Básica (shrinkage) | L1 + L2 + shrinkage | L1 + L2 + shrinkage |
| **Valores faltantes** | Precisa tratar antes | Trata nativamente | Trata nativamente |
| **Features categóricas** | Precisa encoding | Precisa encoding | **Suporte nativo** |
| **GPU** | Não | Sim | Sim |
| **Overfitting em dados pequenos** | Moderado | Moderado | **Maior risco** (leaf-wise) |
| **Quando usar** | Datasets pequenos, baseline | Competições, produção geral | Datasets grandes, alta velocidade |

**Resumo prático:** LightGBM é geralmente a primeira escolha para dados grandes. XGBoost é o mais equilibrado e bem documentado. GradientBoosting do sklearn serve como baseline simples.
"""

NUMERIC_FEATURES = [
  "promised_days",
  "item_count",
  "seller_count",
  "total_price",
  "total_freight"
]

CATEGORICAL_FEATURES = [
  "purchase_month",
  "purchase_weekday",
  "purchase_hour",
  "customer_state"
]

def create_preprocessor() -> ColumnTransformer:
  return ColumnTransformer(
    transformers=[
      # Nas colunas numéricas
      ("numeric", "passthrough", NUMERIC_FEATURES),
      # Nas colunas categoricas
      ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
  )

def create_gradient_boosting_pipeline() -> Pipeline:
  """
  Cria o pipeline do pré-processamento e 
  treinamento do modelo.
  """
  preprocessor = create_preprocessor()

  model = GradientBoostingClassifier(
    n_estimators=100, # número de árvores na floresta.
    learning_rate=0.1, # Taxa de aprendizado
    max_depth=3,
    random_state=42
  )

  return Pipeline(
    steps=[
      ("preprocessor", preprocessor),
      ("model", model)
    ]
  )

def create_xgboost_pipeline() -> Pipeline:
  preprocessor = create_preprocessor()

  model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
  )

  return Pipeline(
    steps=[
      ("preprocessor", preprocessor),
      ("model", model)
    ]
  )

def create_lightgbm_pipeline() -> Pipeline:
  preprocessor = create_preprocessor()

  model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
  )

  return Pipeline(
    steps=[
      ("preprocessor", preprocessor),
      ("model", model)
    ]
  )