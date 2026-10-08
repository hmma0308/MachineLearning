# module_Olist/main.py
import pandas as pd
from pathlib import Path
from loguru import logger

from dataset import load_data, save_data
from features import create_dataset, create_features

# Import substituído para puxar a rotina de treino e salvamento do melhor modelo
from modeling.train import train_and_save_best_model

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

    # 5. Salva o resultado final no interim
    save_data(dataset=dataset_with_features, output_dir=interim_dir)

    logger.success("Pipeline finalizado. Dados salvos em data/interim")

    # 6. Avalia todos os modelos, treina o melhor e salva na pasta models/
    logger.info("Iniciando a etapa de modelagem...")
    train_and_save_best_model(dataset_with_features)

if __name__ == "__main__":
    main()