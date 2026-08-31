import pandas as pd
from pathlib import Path
from loguru import logger

from dataset import load_data, save_data
from features import create_dataset, create_features

# Import adicionado para o novo módulo
from cross_validation import run_cross_validation

def main():
    # Define os caminhos de entrada e saída relativos à raiz do projeto
    raw_dir = Path("data/raw")
    interim_dir = Path("data/interim")

    # Mapeia os arquivos brutos 
    orders_path = raw_dir / "olist_orders_dataset.csv"
    items_path = raw_dir / "olist_order_items_dataset.csv"
    products_path = raw_dir / "olist_products_dataset.csv"
    customers_path = raw_dir / "olist_customers_dataset.csv"

    logger.info("Iniciando o pipeline de processamento de dados...")

    # 1. Carrega as bases
    orders, items, products = load_data(
        order_path=orders_path, 
        items_path=items_path, 
        products_path=products_path
    )

    # 2. Carrega a base de clientes
    logger.info("Carregando o dataset de clientes...")
    customers = pd.read_csv(customers_path)

    # 3. Integra os dados e cria o alvo
    logger.info("Construindo o dataset analítico e criando a variável-alvo...")
    dataset = create_dataset(orders=orders, items=items, customers=customers)

    # 4. Cria as novas variáveis preditivas
    logger.info("Criando as features derivadas...")
    dataset_with_features = create_features(dataset)

    # 5. Salva o resultado final
    save_data(dataset=dataset_with_features, output_dir=interim_dir)

    logger.success("Pipeline finalizado. Dados salvos em data/interim")

    # 6. Executa a validação cruzada no dataset com features
    logger.info("Iniciando a etapa de avaliação de modelo (Cross-Validation)...")
    run_cross_validation(dataset_with_features)


#    data = create_dataset(orders, items, customers)
#    data = create_features(data)

#    # Salva o resultado
#    save_dataset(data, INTERIM_DATA_DIR / "orders_dataset_refined.csv")

#    X_train, X_test, y_train, y_test = split_data(data)

#    models = train_models(X_train, y_train)

#    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()