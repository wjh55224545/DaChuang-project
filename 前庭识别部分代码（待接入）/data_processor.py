# data_processor.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
import re


class VestibularDataset(Dataset):
    """前庭振动特征数据集"""
    
    PARAM_NAMES = [
        'aggression', 'stress', 'tension', 'suspicious', 'balance',
        'confidence', 'vitality', 'self_regulation', 'depression', 'neuroticism'
    ]
    
    PARAM_RANGES = {
        'aggression': (25, 75), 'stress': (15, 65), 'tension': (30, 85),
        'suspicious': (15, 60), 'balance': (30, 85), 'confidence': (40, 90),
        'vitality': (5, 50), 'self_regulation': (40, 90), 'depression': (5, 45),
        'neuroticism': (25, 75)
    }
    
    NORMAL_NORMS = {
        'aggression': 40.5, 'stress': 30.5, 'tension': 64.2, 'suspicious': 32.9,
        'balance': 64.2, 'confidence': 74.9, 'vitality': 22.9,
        'self_regulation': 69.3, 'depression': 16.6, 'neuroticism': 32.0
    }
    
    STANDARDIZATION_FACTORS = {
        'aggression': 1/6.37, 'stress': 1/6.45, 'tension': 1/9.17,
        'suspicious': 1/3.58, 'balance': 1/9.17, 'confidence': 1/8.79,
        'vitality': 1/6.83, 'self_regulation': 1/7.96, 'depression': 1/3.18,
        'neuroticism': 1/11.76
    }
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 param_idx: Optional[int] = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.param_idx = param_idx
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.param_idx is not None:
            return self.features[idx], self.labels[idx, self.param_idx]
        return self.features[idx], self.labels[idx]


class DataProcessor:
    """数据处理器"""
    
    # 默认数据路径
    DEFAULT_DATA_PATH = r"C:\Users\王建豪\Desktop\data"
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据根目录，包含 features/ 和 labels/ 子目录
        """
        if data_dir is None:
            data_dir = self.DEFAULT_DATA_PATH
        
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.labels_dir = self.data_dir / "labels"
        self.scaler = StandardScaler()
        
        # 打印实际路径
        print(f"数据目录: {self.data_dir.absolute()}")
        print(f"特征目录: {self.features_dir.absolute()}")
        print(f"标签目录: {self.labels_dir.absolute()}")
        
    def extract_sample_id(self, filename: str) -> str:
        """
        从文件名提取样本ID（数字部分）
        例如: "1_features.csv" -> "1"
              "3217_features.csv" -> "3217"
        """
        name = Path(filename).stem
        # 使用正则提取数字ID
        match = re.search(r'(\d+)', name)
        if match:
            return match.group(1)
        return name
    
    def _natural_sort_key(self, filepath: str) -> int:
        """自然排序的键函数（按数字大小排序）"""
        match = re.search(r'(\d+)', Path(filepath).stem)
        return int(match.group(1)) if match else 0
    
    def find_feature_files(self) -> List[str]:
        """
        查找所有CSV特征文件，按数字ID自然排序
        """
        print(f"\n正在搜索特征文件...")
        print(f"搜索目录: {self.features_dir.absolute()}")
        
        # 检查目录是否存在
        if not self.features_dir.exists():
            print(f"特征目录不存在，尝试创建...")
            self.features_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        # 列出目录中所有文件
        all_items = list(self.features_dir.iterdir())
        print(f"目录中共有 {len(all_items)} 个项目")
        
        # 显示前10个文件
        print("目录中的文件:")
        for item in all_items[:20]:
            print(f"  - {item.name}")
        if len(all_items) > 20:
            print(f"  ... 等共 {len(all_items)} 个文件")
        
        # 查找所有CSV文件（不区分大小写）
        csv_files = []
        for item in all_items:
            if item.is_file():
                name_lower = item.name.lower()
                if name_lower.endswith('.csv'):
                    csv_files.append(str(item))
                elif 'csv' in name_lower:
                    # 可能是没有.csv扩展名的CSV文件
                    csv_files.append(str(item))
        
        print(f"\n找到 {len(csv_files)} 个CSV文件")
        
        if len(csv_files) == 0:
            print("未找到任何CSV文件！")
            return []
        
        # 显示找到的CSV文件
        print("CSV文件列表:")
        for f in csv_files[:20]:
            print(f"  - {Path(f).name}")
        if len(csv_files) > 20:
            print(f"  ... 等共 {len(csv_files)} 个文件")
        
        # 按数字ID排序（只要文件名中有数字）
        feature_files = []
        for f in csv_files:
            name = Path(f).stem
            if re.search(r'\d+', name):
                feature_files.append(f)
        
        # 按数字ID自然排序
        feature_files = sorted(feature_files, key=self._natural_sort_key)
        
        print(f"\n最终筛选出 {len(feature_files)} 个特征文件")
        if feature_files:
            print("排序后的文件（前10个）:")
            for f in feature_files[:10]:
                print(f"  - {Path(f).name} (ID: {self.extract_sample_id(f)})")
            if len(feature_files) > 10:
                print(f"  ... 等共 {len(feature_files)} 个文件")
        
        return feature_files
    
    def find_labels_file(self) -> Optional[str]:
        """
        查找标签文件
        """
        print(f"\n正在搜索标签文件...")
        print(f"搜索目录: {self.labels_dir.absolute()}")
        
        if not self.labels_dir.exists():
            print(f"标签目录不存在，尝试创建...")
            self.labels_dir.mkdir(parents=True, exist_ok=True)
            return None
        
        # 列出目录中所有文件
        all_items = list(self.labels_dir.iterdir())
        print(f"目录中共有 {len(all_items)} 个项目")
        
        # 查找所有CSV文件
        csv_files = []
        for item in all_items:
            if item.is_file():
                name_lower = item.name.lower()
                if name_lower.endswith('.csv'):
                    csv_files.append(str(item))
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        if not csv_files:
            print("未找到任何标签CSV文件！")
            return None
        
        # 显示找到的CSV文件
        print("CSV文件列表:")
        for f in csv_files:
            print(f"  - {Path(f).name}")
        
        # 优先选择包含parameters的文件
        for f in csv_files:
            name = Path(f).stem.lower()
            if 'parameter' in name and 'confidence' not in name:
                print(f"\n选择标签文件: {Path(f).name}")
                return f
        
        # 其次选择包含dfew的文件
        for f in csv_files:
            name = Path(f).stem.lower()
            if 'dfew' in name and 'confidence' not in name:
                print(f"\n选择标签文件: {Path(f).name}")
                return f
        
        # 返回第一个CSV文件
        print(f"\n选择标签文件: {Path(csv_files[0]).name}")
        return csv_files[0]
    
    def load_features_from_csv(self, csv_path: str) -> pd.DataFrame:
        """从CSV加载特征数据"""
        df = pd.read_csv(csv_path)
        return df
    
    def aggregate_window_features(self, df: pd.DataFrame, 
                                   agg_method: str = 'mean') -> np.ndarray:
        """
        聚合多个窗口的特征
        """
        # 标准特征列名
        standard_cols = ['dominant_freq', 'amp_mean_x', 'amp_mean_y', 
                        'symmetry_ratio', 'motion_stability', 'freq_entropy',
                        'energy_distribution']
        
        # 检查哪些列存在
        available_cols = [col for col in standard_cols if col in df.columns]
        
        if not available_cols:
            # 如果没有标准列名，使用所有数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除可能的时间戳列
            exclude_cols = ['window_id', 'timestamp_sec', 'Unnamed: 0', 'order']
            available_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if not available_cols:
                available_cols = numeric_cols[:7]  # 取前7个
        
        if not available_cols:
            raise ValueError(f"无法找到特征列，数据列: {list(df.columns)}")
        
        if agg_method == 'mean':
            return df[available_cols].mean().values
        elif agg_method == 'median':
            return df[available_cols].median().values
        elif agg_method == 'std':
            return df[available_cols].std().fillna(0).values
        elif agg_method == 'max':
            return df[available_cols].max().values
        elif agg_method == 'min':
            return df[available_cols].min().values
        elif agg_method == 'all':
            mean_vals = df[available_cols].mean().values
            std_vals = df[available_cols].std().fillna(0).values
            max_vals = df[available_cols].max().values
            min_vals = df[available_cols].min().values
            return np.concatenate([mean_vals, std_vals, max_vals, min_vals])
        else:
            raise ValueError(f"不支持的聚合方法: {agg_method}")
    
    def prepare_data(self, agg_method: str = 'mean') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        """
        # 1. 查找特征文件
        feature_files = self.find_feature_files()
        if not feature_files:
            raise ValueError(f"在 {self.features_dir.absolute()} 中未找到CSV特征文件！")
        
        # 2. 查找标签文件
        labels_file = self.find_labels_file()
        if not labels_file:
            raise ValueError(f"在 {self.labels_dir.absolute()} 中未找到标签文件！")
        
        # 3. 加载标签数据
        print(f"\n加载标签数据: {Path(labels_file).name}")
        labels_df = pd.read_csv(labels_file)
        print(f"标签数据形状: {labels_df.shape}")
        print(f"标签列: {list(labels_df.columns)[:15]}...")
        
        # 4. 确定标签列
        label_cols = ['aggression', 'stress', 'tension', 'suspicious', 
                      'balance', 'confidence', 'vitality', 'self_regulation', 
                      'depression', 'neuroticism']
        
        # 找出实际存在的标签列
        available_label_cols = []
        for col in label_cols:
            if col in labels_df.columns:
                available_label_cols.append(col)
        
        if len(available_label_cols) == 0:
            # 尝试使用数值列
            numeric_cols = labels_df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除order列
            if 'order' in labels_df.columns:
                numeric_cols = [c for c in numeric_cols if c != 'order']
            
            if len(numeric_cols) >= 10:
                available_label_cols = numeric_cols[:10]
            else:
                available_label_cols = numeric_cols
        
        print(f"使用标签列 ({len(available_label_cols)}个): {available_label_cols}")
        
        # 5. 构建ID到标签的映射
        id_to_labels = {}
        use_order_mapping = False
        
        if 'order' in labels_df.columns:
            for _, row in labels_df.iterrows():
                if pd.notna(row['order']):
                    sample_id = str(int(row['order']))
                    labels = row[available_label_cols].values.astype(np.float32)
                    id_to_labels[sample_id] = labels
            use_order_mapping = True
            print(f"通过'order'列建立了 {len(id_to_labels)} 个标签映射")
        
        # 6. 处理每个特征文件
        all_features = []
        all_labels = []
        file_ids = []
        matched_count = 0
        unmatched_ids = []
        
        print(f"\n处理特征文件并匹配标签...")
        print(f"使用聚合方法: {agg_method}")
        
        for i, feat_path in enumerate(feature_files):
            file_id = self.extract_sample_id(feat_path)
            
            try:
                # 加载特征
                df = self.load_features_from_csv(feat_path)
                features = self.aggregate_window_features(df, agg_method)
                
                # 查找标签
                if use_order_mapping and file_id in id_to_labels:
                    label_values = id_to_labels[file_id]
                    matched_count += 1
                elif use_order_mapping:
                    unmatched_ids.append(file_id)
                    continue
                else:
                    # 没有ID映射，按顺序匹配
                    idx = i % len(labels_df)
                    label_values = labels_df.iloc[idx][available_label_cols].values.astype(np.float32)
                    matched_count += 1
                
                all_features.append(features)
                all_labels.append(label_values)
                file_ids.append(file_id)
                
                # 显示进度
                if matched_count <= 5 or matched_count % 500 == 0:
                    print(f"  ✓ [{matched_count}] ID:{file_id} -> 特征维度={len(features)}, "
                          f"标签示例=[{label_values[0]:.2f}, {label_values[1]:.2f}, {label_values[2]:.2f}]")
                
            except Exception as e:
                print(f"  ✗ 处理 {file_id} 失败: {e}")
                continue
        
        # 7. 输出匹配统计
        print(f"\n" + "="*50)
        print(f"数据匹配统计")
        print("="*50)
        print(f"特征文件总数: {len(feature_files)}")
        print(f"成功匹配: {matched_count}")
        
        if unmatched_ids:
            print(f"未匹配: {len(unmatched_ids)}")
            if len(unmatched_ids) <= 20:
                print(f"  未匹配ID: {unmatched_ids}")
        
        if not all_features:
            raise ValueError("没有成功加载任何数据！")
        
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        print(f"\n最终数据集:")
        print(f"  特征形状: {features_array.shape}")
        print(f"  标签形状: {labels_array.shape}")
        
        # 8. 数据质量检查
        self._check_data_quality(features_array, labels_array, available_label_cols)
        
        return features_array, labels_array, file_ids
    
    def _check_data_quality(self, features: np.ndarray, labels: np.ndarray, 
                            label_names: List[str]):
        """检查数据质量"""
        print("\n" + "="*60)
        print("数据质量检查")
        print("="*60)
        
        # 检查特征
        print("\n特征统计:")
        for i in range(min(7, features.shape[1])):
            feat_mean = np.mean(features[:, i])
            feat_std = np.std(features[:, i])
            feat_min = np.min(features[:, i])
            feat_max = np.max(features[:, i])
            print(f"  特征{i}: 均值={feat_mean:.4f}, 标准差={feat_std:.4f}, "
                  f"范围=[{feat_min:.4f}, {feat_max:.4f}]")
        
        # 检查标签统计
        print("\n标签统计:")
        for i, name in enumerate(label_names):
            if i < labels.shape[1]:
                mean_val = np.mean(labels[:, i])
                std_val = np.std(labels[:, i])
                min_val = np.min(labels[:, i])
                max_val = np.max(labels[:, i])
                print(f"  {name:15s}: 均值={mean_val:6.2f}, 标准差={std_val:5.2f}, "
                      f"范围=[{min_val:.1f}, {max_val:.1f}]")
        
        # 检查特征-标签相关性
        print("\n特征与标签的平均相关系数（绝对值）:")
        correlations = []
        for i in range(min(7, features.shape[1])):
            corr_sum = 0
            for j in range(min(10, labels.shape[1])):
                if np.std(features[:, i]) > 0 and np.std(labels[:, j]) > 0:
                    corr = np.corrcoef(features[:, i], labels[:, j])[0, 1]
                    corr_sum += abs(corr)
            avg_corr = corr_sum / min(10, labels.shape[1])
            correlations.append(avg_corr)
            print(f"  特征{i}: {avg_corr:.4f}")
        
        avg_total_corr = np.mean(correlations) if correlations else 0
        print(f"\n平均特征-标签相关性: {avg_total_corr:.4f}")
        
        if avg_total_corr < 0.1:
            print("⚠️ 警告: 特征与标签相关性较低")
        elif avg_total_corr < 0.2:
            print("△ 特征与标签有一定相关性")
        else:
            print("✓ 特征与标签相关性较好")
    
    def split_data(self, features: np.ndarray, labels: np.ndarray,
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42) -> Dict[str, Tuple]:
        """划分训练集、验证集、测试集"""
        n_samples = len(features)
        print(f"\n划分数据集 (总样本: {n_samples})")
        
        # 划分训练+验证和测试
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )
        
        # 划分训练和验证
        if val_size > 0 and len(X_train_val) > 1:
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, 
                random_state=random_state
            )
        else:
            X_train, X_val = X_train_val, np.array([])
            y_train, y_val = y_train_val, np.array([])
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val) if len(X_val) > 0 else np.array([])
        
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        return {
            'train': (X_train_scaled, y_train),
            'val': (X_val_scaled, y_val),
            'test': (X_test_scaled, y_test)
        }