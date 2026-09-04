import copy
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import wmae_score_numpy


def train_model(model: nn.Module, epochs: int, train_loader: DataLoader, test_loader: DataLoader, scaler_Y: StandardScaler) -> nn.Module:
    """Treina o modelo a partir do DataLoader de treino"""
    criterion_mse = nn.MSELoss()    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    best_test_wmae = float('inf') 
    best_model_weights = None
    best_epoch = 1
    
    for epoch in range(epochs):
        model.train()                
        train_predictions = []
        train_targets = []
        train_holidays = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # move batches to same device as model
            device = next(model.parameters()).device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)
          
            loss_mse = criterion_mse(predictions, batch_y)
            loss_mse.backward()
            optimizer.step()
            
            train_predictions.extend(predictions.detach().cpu().numpy().ravel().tolist())
            train_targets.extend(batch_y.detach().cpu().numpy().ravel().tolist())
            train_holidays.extend(batch_X[:, -1].detach().cpu().numpy().ravel().tolist())

        train_preds_np = np.array(train_predictions).reshape(-1, 1)
        train_targets_np = np.array(train_targets).reshape(-1, 1)
        train_holidays_np = np.array(train_holidays).reshape(-1, 1)

        train_preds_real = scaler_Y.inverse_transform(train_preds_np)
        train_targets_real = scaler_Y.inverse_transform(train_targets_np)

        avg_train_rmse = mean_squared_error(train_targets_real, train_preds_real) ** 0.5
        avg_train_mae = mean_absolute_error(train_targets_real, train_preds_real)
        train_r2 = r2_score(train_targets_real, train_preds_real)

        avg_train_wmae = wmae_score_numpy(train_targets_real, train_preds_real, train_holidays_np)

        model.eval()
                
        test_predictions = []
        test_targets = []
        test_holidays = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                device = next(model.parameters()).device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X)
                
                test_predictions.extend(predictions.detach().cpu().numpy().ravel().tolist())
                test_targets.extend(batch_y.detach().cpu().numpy().ravel().tolist())
                test_holidays.extend(batch_X[:, -1].detach().cpu().numpy().ravel().tolist())
                
        test_preds_np = np.array(test_predictions).reshape(-1, 1)
        test_targets_np = np.array(test_targets).reshape(-1, 1)
        test_holidays_np = np.array(test_holidays).reshape(-1, 1)
        
        test_preds_real = scaler_Y.inverse_transform(test_preds_np)
        test_targets_real = scaler_Y.inverse_transform(test_targets_np)

        avg_test_rmse = mean_squared_error(test_targets_real, test_preds_real) ** 0.5
        avg_test_mae = mean_absolute_error(test_targets_real, test_preds_real)
        test_r2 = r2_score(test_targets_real, test_preds_real)
        avg_test_wmae = wmae_score_numpy(test_targets_real, test_preds_real, test_holidays_np)

        if avg_test_wmae < best_test_wmae:
            best_test_wmae = avg_test_wmae         
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  TRAIN -> RMSE: {avg_train_rmse:.2f} | MAE: {avg_train_mae:.2f} | WMAE: {avg_train_wmae:.2f} | R2: {train_r2:.4f}")
        print(f"  TEST  -> RMSE: {avg_test_rmse:.2f} | MAE: {avg_test_mae:.2f} | WMAE: {avg_test_wmae:.2f} | R2: {test_r2:.4f}")
        print("-" * 50)
    model.load_state_dict(best_model_weights)
    print(f"Treino concluído! Melhor modelo resgatado da época {best_epoch} com WMAE de {best_test_wmae:.2f}")
    return model, best_test_wmae

