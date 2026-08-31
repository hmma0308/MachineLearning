import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Remove colunas que não podem ser usadas no treinamento (IDs, datas e data leakage)
    e separa a variável-alvo (y) das variáveis preditivas (X).
    """
    logger.info("Preparando dados: removendo IDs e colunas que causam vazamento de dados...")
    
    # Colunas que causam data leakage ou não são interpretáveis matematicamente
    columns_to_drop = [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "is_late"  # A própria variável-alvo
    ]
    
    # Removendo customer_city devido à altíssima cardinalidade para um modelo inicial.
    # Sugestão futura: usar target encoding para customer_city se quiser mantê-la.
    if "customer_city" in df.columns:
        columns_to_drop.append("customer_city")
        
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    y = df["is_late"]
    
    return X, y

def build_pipeline() -> Pipeline:
    """
    Constrói um pipeline isolado com processamento de dados e o classificador.
    """
    logger.info("Construindo o pipeline de Machine Learning...")
    
    # Separando os tipos de features geradas no features.py e dataset.py
    numeric_features = [
        "item_count", "seller_count", "total_price", "total_freight", 
        "promised_days", "purchase_month", "purchase_weekday", "purchase_hour"
    ]
    categorical_features = ["customer_state"]
    
    # Tratamento numérico: preenche nulos com mediana e padroniza a escala
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Tratamento categórico: preenche nulos com a moda e faz One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Combina os processadores nas colunas corretas
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])
    
    # Instancia o modelo. O class_weight="balanced" ajuda pois a maioria das entregas não atrasa.
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    
    # Monta o pipeline final
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    return pipeline

def run_cross_validation(df: pd.DataFrame, cv_folds: int = 5):
    """
    Executa a validação cruzada no dataset usando métricas variadas.
    """
    X, y = prepare_data(df)
    pipeline = build_pipeline()
    
    logger.info(f"Iniciando Cross-Validation com {cv_folds} folds (processo pode levar alguns segundos)...")
    
    # Focando em métricas essenciais para desbalanceamento de classes
    scoring = ["accuracy", "roc_auc", "f1"]
    
    cv_results = cross_validate(
        pipeline, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1
    )
    
    logger.success("Cross-Validation concluída com sucesso.")
    logger.info(f"Média Accuracy: {cv_results['test_accuracy'].mean():.4f}")
    logger.info(f"Média ROC-AUC:  {cv_results['test_roc_auc'].mean():.4f}")
    logger.info(f"Média F1-Score: {cv_results['test_f1'].mean():.4f}")
    
    return cv_results