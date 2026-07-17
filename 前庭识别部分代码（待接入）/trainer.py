# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_processor import VestibularDataset, DataProcessor
from models import (
    BaseParameterModel, MultiTaskVestibularModel, AttentionFusionModel,
    AggressionModel, StressModel, TensionModel, SuspiciousModel,
    BalanceModel, ConfidenceModel, VitalityModel, SelfRegulationModel,
    DepressionModel, NeuroticismModel
)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ParameterModelTrainer:
    """单参数模型训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 150, patience: int = 15, 
              model_save_path: Optional[str] = None) -> Dict:
        """完整训练流程"""
        early_stopping = EarlyStopping(patience=patience)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss
                    }, model_save_path)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }


class MultiTaskTrainer:
    """多任务模型训练器"""
    
    def __init__(self, model: MultiTaskVestibularModel, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.param_losses = {name: [] for name in VestibularDataset.PARAM_NAMES}
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        param_losses = {name: 0 for name in VestibularDataset.PARAM_NAMES}
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # 计算各参数损失
            loss = 0
            for i, param_name in enumerate(VestibularDataset.PARAM_NAMES):
                param_loss = self.criterion(predictions[param_name], labels[:, i])
                loss += param_loss
                param_losses[param_name] += param_loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_total_loss = total_loss / len(train_loader)
        avg_param_losses = {k: v / len(train_loader) for k, v in param_losses.items()}
        
        return avg_total_loss, avg_param_losses
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """验证"""
        self.model.eval()
        total_loss = 0
        param_losses = {name: 0 for name in VestibularDataset.PARAM_NAMES}
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(features)
                
                loss = 0
                for i, param_name in enumerate(VestibularDataset.PARAM_NAMES):
                    param_loss = self.criterion(predictions[param_name], labels[:, i])
                    loss += param_loss
                    param_losses[param_name] += param_loss.item()
                
                total_loss += loss.item()
                
        avg_total_loss = total_loss / len(val_loader)
        avg_param_losses = {k: v / len(val_loader) for k, v in param_losses.items()}
        
        return avg_total_loss, avg_param_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 150, patience: int = 15,
              model_save_path: Optional[str] = None) -> Dict:
        """完整训练流程"""
        early_stopping = EarlyStopping(patience=patience)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss, train_param_losses = self.train_epoch(train_loader)
            val_loss, val_param_losses = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            for name in VestibularDataset.PARAM_NAMES:
                self.param_losses[name].append(val_param_losses[name])
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss
                    }, model_save_path)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
            
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'param_losses': self.param_losses,
            'best_val_loss': best_val_loss
        }


class ModelFactory:
    """模型工厂类"""
    
    def __init__(self, input_dim: int = 7, device: str = 'cuda'):
        self.input_dim = input_dim
        self.device = device
        self.models = {}
        self.trainers = {}
        
        self._initialize_models()
        
    def _initialize_models(self):
        """初始化所有十个模型"""
        model_classes = [
            AggressionModel, StressModel, TensionModel, SuspiciousModel,
            BalanceModel, ConfidenceModel, VitalityModel, SelfRegulationModel,
            DepressionModel, NeuroticismModel
        ]
        
        for model_cls in model_classes:
            model = model_cls(self.input_dim)
            param_name = model.param_name
            self.models[param_name] = model
            self.trainers[param_name] = ParameterModelTrainer(model, self.device)
    
    def train_all(self, data_splits: Dict, epochs: int = 150,
                  save_dir: str = "models") -> Dict:
        """训练所有模型"""
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        
        for i, param_name in enumerate(VestibularDataset.PARAM_NAMES):
            print(f"\n{'='*50}")
            print(f"Training {param_name} model")
            print('='*50)
            
            # 创建数据加载器
            train_dataset = VestibularDataset(X_train, y_train, param_idx=i)
            val_dataset = VestibularDataset(X_val, y_val, param_idx=i)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # 训练
            save_path = os.path.join(save_dir, f"{param_name}_model.pt")
            result = self.trainers[param_name].train(
                train_loader, val_loader, epochs=epochs,
                model_save_path=save_path
            )
            
            results[param_name] = result
            
        return results
    
    def predict_all(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """预测所有参数"""
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        predictions = {}
        for param_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                pred = model(features_tensor)
                predictions[param_name] = pred.cpu().numpy()
                
        return predictions
    
    def calculate_K_value(self, predictions: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """计算整合系数K"""
        K = 0
        deviations = {}
        
        for param_name, measured_value in predictions.items():
            norm_value = VestibularDataset.NORMAL_NORMS[param_name]
            m = VestibularDataset.STANDARDIZATION_FACTORS[param_name]
            
            deviation = m * (measured_value - norm_value)
            deviations[param_name] = deviation
            K += deviation
            
        return K, deviations
    
    def interpret_K(self, K: float) -> str:
        """解释K值"""
        if abs(K) < 3:
            return "情绪状态稳定，接近正常常模水平"
        elif abs(K) < 6:
            return "轻度偏离常模，建议关注情绪变化"
        else:
            return "显著偏离常模，建议进行专业情绪干预"
    
    def save_all_models(self, save_dir: str = "models"):
        """保存所有模型"""
        os.makedirs(save_dir, exist_ok=True)
        for param_name, model in self.models.items():
            save_path = os.path.join(save_dir, f"{param_name}_model.pt")
            torch.save(model.state_dict(), save_path)
        print(f"All models saved to {save_dir}")
    
    def load_all_models(self, save_dir: str = "models"):
        """加载所有模型"""
        for param_name, model in self.models.items():
            save_path = os.path.join(save_dir, f"{param_name}_model.pt")
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path, map_location=self.device))
                print(f"Loaded {param_name} model")
            else:
                print(f"Warning: {param_name} model not found at {save_path}")