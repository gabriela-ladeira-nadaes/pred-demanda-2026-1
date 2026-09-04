import pandas as pd
import torch
import copy

from data.data_loader import load_data, validate_data
from data.datasets import get_device, to_tensors, make_dataset, make_dataloader
from preprocessing.transform import clean_data, split_data, standardize, describe_array, standardize_y
from preprocessing.features import create_features
from utils.config import DATA_PATH, BATCH_SIZE, FIXED_SEED, EPOCHS
from training.train import train_model
from models.model import FinnancialModel,LinearRegression, save_model

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
    # Split cronologico e normalizacao
    X_train,y_train, X_test, y_test = split_data(data)
    X_train, X_test = standardize(X_train, X_test)

    #Normalizar Y
    y_train_scaled, y_test_scaled, scaler_y = standardize_y(y_train,y_test)

    print("Estatísticas do alvo (y_train):", describe_array(y_train))
    print("Estatísticas do alvo (y_test):", describe_array(y_test))
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # Criacao de tensores, dataset e dataloader
    device = get_device()
    X_train_t, y_train_t = to_tensors(X_train, y_train_scaled)
    X_test_t, y_test_t = to_tensors(X_test, y_test_scaled)

    train_dataset = make_dataset(X_train_t, y_train_t)
    test_dataset = make_dataset(X_test_t, y_test_t)
    train_loader = make_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
    test_loader = make_dataloader(test_dataset, batch_size=BATCH_SIZE)

    torch.manual_seed(FIXED_SEED)
    input_dim = X_train_t.shape[1]
    output_dim = 1  # Dimensao de saida

    models = {
    "Linear Regression": LinearRegression(input_dim, output_dim),
    "Multilayer Perceptron": FinnancialModel(input_dim, output_dim)
    }

    best_wmae_global = float('inf')
    best_modelo_global = None
    best_name = ""
    
    for model_name, model in models.items():
        print(f"\n{'='*20} Iniciando teste: {model_name} {'='*20}")        
        model = model.to(device)        
        # Recebe o modelo treinado e a pontuação dele
        model_trained, wmae_result = train_model(model, EPOCHS, train_loader, test_loader, scaler_y)
        
        print(f"[{model_name}] finalizou com WMAE: {wmae_result:.2f}")        
       
        if wmae_result < best_wmae_global:
            best_wmae_global = wmae_result
            best_name = model_name            
            best_modelo_global = copy.deepcopy(model_trained)
   
    print(f"\nO Modelo escolhido foi '{best_name}' com um WMAE de ${best_wmae_global:.2f}!")
    save_model(best_modelo_global)   
   
if __name__ == "__main__":
    main()
