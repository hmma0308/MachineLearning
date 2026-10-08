from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

from module_olist.config import FIGURES_DIR, INTERIM_DATA_DIR, REPORTS_DIR
from module_olist.modeling.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from module_olist.modeling.split import FEATURES, TARGET, split_data

RANDOM_STATE = 42

DATASET_PATH = INTERIM_DATA_DIR / "dataset.csv"
HISTORY_PATH = REPORTS_DIR / "active_learning.csv"
FIGURE_PATH = FIGURES_DIR / "active_learning.png"

POOL_SIZE = 5_000

N_INITIAL_LABELED = 100

N_QUERIES_PER_ITERATION = 50

MAX_ITERATIONS = 20

UNLABELED = -1

STRATEGIES = ("uncertainty", "random")


def create_label_spreading_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LabelSpreading(
                    kernel="knn",
                    n_neighbors=10,
                    alpha=0.2,
                    max_iter=50,
                ),
            ),
        ]
    )


def select_queries(
    model: Pipeline,
    unlabeled_indices: np.ndarray,
    n_queries: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Escolhe quais pedidos vão ser revelados na próxima rodada."""
    n_queries = min(n_queries, len(unlabeled_indices))

    if strategy == "random":
        return rng.choice(unlabeled_indices, size=n_queries, replace=False)

    if strategy != "uncertainty":
        raise ValueError(
            f"Estratégia bizarra: {strategy}. Usa uma dessas: {STRATEGIES}.")

    distributions = model.named_steps["model"].label_distributions_[
        unlabeled_indices]
    entropies = stats.entropy(np.nan_to_num(distributions).T)

    most_uncertain = np.argsort(entropies)[::-1][:n_queries]

    return unlabeled_indices[most_uncertain]


def run_active_learning(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: str = "uncertainty",
    n_initial_labeled: int = N_INITIAL_LABELED,
    n_queries: int = N_QUERIES_PER_ITERATION,
    max_iterations: int = MAX_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    X_pool = X_pool.reset_index(drop=True)
    y_pool = y_pool.to_numpy()
    rng = np.random.default_rng(random_state)

    initial_indices, _ = train_test_split(
        np.arange(len(y_pool)),
        train_size=n_initial_labeled,
        stratify=y_pool,
        random_state=random_state,
    )

    labeled = np.zeros(len(y_pool), dtype=bool)
    labeled[initial_indices] = True

    history = []

    for iteration in range(max_iterations + 1):
        y_train = np.where(labeled, y_pool, UNLABELED)

        model = create_label_spreading_pipeline()
        model.fit(X_pool, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        history.append(
            {
                "strategy": strategy,
                "iteration": iteration,
                "n_labeled": int(labeled.sum()),
                "n_positive_labeled": int(y_pool[labeled].sum()),
                "pr_auc": float(average_precision_score(y_test, y_proba)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }
        )

        logger.info(
            "{} | rodada {:>2} | rótulos: {:>5} ({} atrasos) | pr_auc: {:.4f} | roc_auc: {:.4f}",
            strategy,
            iteration,
            history[-1]["n_labeled"],
            history[-1]["n_positive_labeled"],
            history[-1]["pr_auc"],
            history[-1]["roc_auc"],
        )

        unlabeled_indices = np.flatnonzero(~labeled)
        if iteration == max_iterations or len(unlabeled_indices) == 0:
            break

        queries = select_queries(
            model, unlabeled_indices, n_queries, strategy, rng)
        labeled[queries] = True

    return pd.DataFrame(history)


def plot_learning_curves(history: pd.DataFrame, baseline: float, path=FIGURE_PATH) -> None:
    """Gera o gráfico comparando o quão rápido as estratégias aprendem."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, group in history.groupby("strategy"):
        ax.plot(group["n_labeled"], group["pr_auc"],
                marker="o", label=strategy)

    ax.axhline(baseline, color="gray", linestyle="--", label="acaso")
    ax.set_xlabel("Pedidos rotulados")
    ax.set_ylabel("PR-AUC no teste")
    ax.set_title("Active learning com LabelSpreading")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.success("Curvas salvas em {}", path)


def main() -> None:
    """Orquestrador do rolê do active learning."""
    logger.info("Lendo a base intermediária de {}", DATASET_PATH)
    dataset = pd.read_csv(DATASET_PATH, usecols=FEATURES + [TARGET])

    X_train, X_test, y_train, y_test = split_data(dataset)

    X_pool, _, y_pool, _ = train_test_split(
        X_train,
        y_train,
        train_size=min(POOL_SIZE, len(X_train)),
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    logger.info("Pool: {} pedidos | Teste: {} pedidos",
                len(X_pool), len(X_test))

    history = pd.concat(
        [
            run_active_learning(X_pool, y_pool, X_test,
                                y_test, strategy=strategy)
            for strategy in STRATEGIES
        ],
        ignore_index=True,
    )

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_PATH, index=False)
    logger.success("Histórico salvo em {}", HISTORY_PATH)

    plot_learning_curves(history, baseline=float(y_test.mean()))


if __name__ == "__main__":
    main()
