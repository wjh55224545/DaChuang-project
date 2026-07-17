# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional,Dict

class BaseParameterModel(nn.Module):
    """所有参数预测模型的基础类"""
    
    def __init__(self, input_dim: int = 7, hidden_dims: List[int] = [64, 32],
                 param_range: Tuple[float, float] = (0, 100),
                 dropout_rate: float = 0.2):
        super().__init__()
        self.param_range = param_range
        self.min_val, self.max_val = param_range
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        raw_output = self.network(x)
        # 将输出约束在合理范围内
        output = torch.sigmoid(raw_output) * (self.max_val - self.min_val) + self.min_val
        return output.squeeze()


class AggressionModel(BaseParameterModel):
    """攻击性参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (25.0, 75.0))
        self.param_name = "aggression"


class StressModel(BaseParameterModel):
    """压力参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (15.0, 65.0))
        self.param_name = "stress"


class TensionModel(BaseParameterModel):
    """紧张参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (30.0, 85.0))
        self.param_name = "tension"


class SuspiciousModel(BaseParameterModel):
    """可疑参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (15.0, 60.0))
        self.param_name = "suspicious"


class BalanceModel(BaseParameterModel):
    """平衡参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (30.0, 85.0))
        self.param_name = "balance"


class ConfidenceModel(BaseParameterModel):
    """自信参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (40.0, 90.0))
        self.param_name = "confidence"


class VitalityModel(BaseParameterModel):
    """活力参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (5.0, 50.0))
        self.param_name = "vitality"


class SelfRegulationModel(BaseParameterModel):
    """自我调节参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (40.0, 90.0))
        self.param_name = "self_regulation"


class DepressionModel(BaseParameterModel):
    """抑郁参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (5.0, 45.0))
        self.param_name = "depression"


class NeuroticismModel(BaseParameterModel):
    """神经质参数预测模型"""
    def __init__(self, input_dim: int = 7):
        super().__init__(input_dim, [64, 32], (25.0, 75.0))
        self.param_name = "neuroticism"


class MultiTaskVestibularModel(nn.Module):
    """
    多任务学习模型
    共享特征提取层，分别预测十个参数
    """
    
    def __init__(self, input_dim: int = 7, shared_dims: List[int] = [64, 32],
                 task_dims: List[int] = [16], dropout_rate: float = 0.2):
        super().__init__()
        
        # 共享特征提取层
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        self.shared_dim = prev_dim
        
        # 各参数专用层
        self.param_heads = nn.ModuleDict({
            'aggression': self._create_head((25.0, 75.0), task_dims, dropout_rate),
            'stress': self._create_head((15.0, 65.0), task_dims, dropout_rate),
            'tension': self._create_head((30.0, 85.0), task_dims, dropout_rate),
            'suspicious': self._create_head((15.0, 60.0), task_dims, dropout_rate),
            'balance': self._create_head((30.0, 85.0), task_dims, dropout_rate),
            'confidence': self._create_head((40.0, 90.0), task_dims, dropout_rate),
            'vitality': self._create_head((5.0, 50.0), task_dims, dropout_rate),
            'self_regulation': self._create_head((40.0, 90.0), task_dims, dropout_rate),
            'depression': self._create_head((5.0, 45.0), task_dims, dropout_rate),
            'neuroticism': self._create_head((25.0, 75.0), task_dims, dropout_rate)
        })
        
    def _create_head(self, param_range: Tuple[float, float], 
                     hidden_dims: List[int], dropout_rate: float) -> nn.Module:
        """创建参数预测头"""
        layers = []
        prev_dim = self.shared_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.ModuleDict({
            'network': nn.Sequential(*layers),
            'min_val': param_range[0],
            'max_val': param_range[1]
        })
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        shared_features = self.shared_network(x)
        
        outputs = {}
        for param_name, head in self.param_heads.items():
            raw_output = head['network'](shared_features)
            output = torch.sigmoid(raw_output) * (head['max_val'] - head['min_val']) + head['min_val']
            outputs[param_name] = output.squeeze()
        
        return outputs


class AttentionFusionModel(nn.Module):
    """
    带注意力机制的双模态融合模型
    支持微表情特征和前庭振动特征的融合
    """
    
    def __init__(self, vestibular_dim: int = 7, micro_expression_dim: int = 675,
                 fusion_dim: int = 128, param_dims: List[int] = [64, 32]):
        super().__init__()
        
        # 前庭振动特征编码器
        self.vestibular_encoder = nn.Sequential(
            nn.Linear(vestibular_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        # 微表情特征编码器（如果有）
        self.micro_encoder = nn.Sequential(
            nn.Linear(micro_expression_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # 融合层
        fusion_input_dim = 32 + 128
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU()
        )
        
        # 参数预测头
        self.param_heads = nn.ModuleDict({
            'aggression': self._create_head(fusion_dim // 2, (25.0, 75.0)),
            'stress': self._create_head(fusion_dim // 2, (15.0, 65.0)),
            'tension': self._create_head(fusion_dim // 2, (30.0, 85.0)),
            'suspicious': self._create_head(fusion_dim // 2, (15.0, 60.0)),
            'balance': self._create_head(fusion_dim // 2, (30.0, 85.0)),
            'confidence': self._create_head(fusion_dim // 2, (40.0, 90.0)),
            'vitality': self._create_head(fusion_dim // 2, (5.0, 50.0)),
            'self_regulation': self._create_head(fusion_dim // 2, (40.0, 90.0)),
            'depression': self._create_head(fusion_dim // 2, (5.0, 45.0)),
            'neuroticism': self._create_head(fusion_dim // 2, (25.0, 75.0))
        })
        
    def _create_head(self, input_dim: int, param_range: Tuple[float, float]) -> nn.Module:
        """创建参数预测头"""
        return nn.ModuleDict({
            'network': nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ),
            'min_val': param_range[0],
            'max_val': param_range[1]
        })
    
    def forward(self, vestibular_features: torch.Tensor, 
                micro_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            vestibular_features: 前庭振动特征 [batch_size, 7]
            micro_features: 微表情特征 [batch_size, 675]，可选
        """
        # 编码前庭特征
        v_encoded = self.vestibular_encoder(vestibular_features)
        
        if micro_features is not None:
            # 编码微表情特征
            m_encoded = self.micro_encoder(micro_features)
            
            # 跨模态注意力
            m_attended, _ = self.cross_attention(m_encoded.unsqueeze(1), 
                                                  m_encoded.unsqueeze(1), 
                                                  m_encoded.unsqueeze(1))
            m_attended = m_attended.squeeze(1)
            
            # 融合
            fused = torch.cat([v_encoded, m_attended], dim=1)
        else:
            # 只有前庭特征时，用零填充微表情部分
            fused = torch.cat([v_encoded, torch.zeros(v_encoded.size(0), 128).to(v_encoded.device)], dim=1)
        
        fused_features = self.fusion_layer(fused)
        
        outputs = {}
        for param_name, head in self.param_heads.items():
            raw_output = head['network'](fused_features)
            output = torch.sigmoid(raw_output) * (head['max_val'] - head['min_val']) + head['min_val']
            outputs[param_name] = output.squeeze()
        
        return outputs