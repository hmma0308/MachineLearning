import numpy as np
from loguru import logger
from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             )

def evaluate_models(
    models, X_test, y_test
):
  for name, model in models.items():
      y_proba = model.predict_proba(X_test)

      best_threshold = None
      best_f1 = -1
      best_precision = None
      best_recall = None

      for threshold in np.range(0.05, 0.50, 0.01):
         y_pred = (y_proba >= threshold).astype(int)

         f1 = f1_score(y_test, y_pred)
         precision = precision_score(y_test, y_pred)
         recall = recall_score(y_test, y_pred)

         if f1 > best_f1:
             best_f1 = f1
             best_threshold = threshold
             best_precision = precision
             best_recall = recall
            
      roc_auc = roc_auc_score(y_test, y_proba,)

      logger.info(f"Modelo: {name}")
      logger.info(f"Melhor threshold: {best_threshold:.2f}")
      logger.info(f"Precision: {best_precision:.3f}")
      logger.info(f"Recall: {best_recall:.3f}")
      logger.info(f"F1: {best_f1:.4f}")
      logger.info(f"ROC-AUC: {roc_auc:.3f}")