import pandas as pd

from data.data_loader import load_data, validate_data
from data.datasets import get_device, to_tensors, make_dataset, make_dataloader
from preprocessing.transform import clean_data, build_features_matrix, split_data, standardize, describe_array
from preprocessing.features import create_features
from utils.config import DATA_PATH

def main() -> None:
    # Carregamento e validacao
    data: pd.DataFrame = load_data(DATA_PATH)
    if not validate_data(data):
        print("Base de dados vazia!")
        return
    print(f"Dados carregados: {data.shape[0]} linhas, {data.shape[1]} colunas.")
    
    # Limpeza e definicao de atributos temporais
    data = clean_data(data)
    data = create_features(data)

    # Criacao de matriz de features e vetor alvo em NumPy
    X, y = build_features_matrix(data)
    print("Estatísticas do alvo (y):", describe_array(y))

    # Split cronologico e normalizacao
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = standardize(X_train, X_test)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # Criacao de tensores, dataset e dataloader
    device = get_device()
    X_train_t, y_train_t = to_tensors(X_train, y_train)
    X_test_t, y_test_t = to_tensors(X_test, y_test)

    train_dataset = make_dataset(X_train_t, y_train_t)
    test_dataset = make_dataset(X_test_t, y_test_t)
    train_loader = make_dataloader(train_dataset, shuffle = True)
    test_loader = make_dataloader(test_dataset)

    X_batch, y_batch = next(iter(train_loader))

    print(f"Device: {device}")
    print(f"Tensores -> X: {X_train_t.shape} {X_train_t.dtype} , y: {y_train_t.shape} {y_train_t.dtype}")
    print(f"Batch de treino -> X: {tuple(X_batch.shape)}, y: {tuple(y_batch.shape)}")
    print(f"Nº de batches (treino): {len(train_loader)} vs. (teste): {len(test_loader)}")

if __name__ == "__main__":
    main()
