# train.py
"""
前庭振动情绪识别系统 - 训练脚本
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime

from data_processor import DataProcessor, VestibularDataset
from trainer import ModelFactory


# 默认数据路径
DEFAULT_DATA_PATH = r"C:\Users\王建豪\Desktop\神经网络\data"


def print_data_summary(features, labels, file_ids):
    """打印数据摘要"""
    print("\n" + "="*60)
    print("数据摘要")
    print("="*60)
    print(f"样本数量: {len(features)}")
    print(f"特征维度: {features.shape[1]}")
    print(f"标签维度: {labels.shape[1]}")
    
    if len(file_ids) <= 20:
        print(f"文件ID: {', '.join(file_ids)}")
    else:
        print(f"文件ID示例: {', '.join(file_ids[:10])} ... 共 {len(file_ids)} 个")


def train_models(data_splits, save_dir="models", epochs=100):
    """训练模型"""
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    X_train, y_train = data_splits['train']
    X_val, y_val = data_splits['val']
    X_test, y_test = data_splits['test']
    
    input_dim = X_train.shape[1]
    
    print(f"输入特征维度: {input_dim}")
    print(f"训练样本数: {len(X_train)}")
    print(f"验证样本数: {len(X_val)}")
    print(f"测试样本数: {len(X_test)}")
    print(f"训练轮数: {epochs}")
    
    # 创建模型工厂
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    factory = ModelFactory(input_dim=input_dim, device=device)
    
    # 训练
    results = factory.train_all(data_splits, epochs=epochs, save_dir=save_dir)
    
    # 保存模型
    factory.save_all_models(save_dir)
    
    return factory, results


def evaluate_models(factory, X_test, y_test):
    """评估模型"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print("\n" + "="*60)
    print("模型评估结果")
    print("="*60)
    
    predictions = factory.predict_all(X_test)
    
    results = {}
    param_names = VestibularDataset.PARAM_NAMES
    
    for i, param_name in enumerate(param_names):
        if i < y_test.shape[1]:
            pred = predictions[param_name]
            true = y_test[:, i]
            
            mae = mean_absolute_error(true, pred)
            rmse = np.sqrt(mean_squared_error(true, pred))
            r2 = r2_score(true, pred)
            
            results[param_name] = {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}
            
            norm_val = VestibularDataset.NORMAL_NORMS[param_name]
            relative_mae = mae / norm_val * 100
            
            print(f"\n{param_name.upper()} (常模={norm_val:.1f}):")
            print(f"  MAE: {mae:.4f} ({relative_mae:.2f}%)")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
    
    if results:
        avg_mae = np.mean([r['MAE'] for r in results.values()])
        avg_rmse = np.mean([r['RMSE'] for r in results.values()])
        avg_r2 = np.mean([r['R2'] for r in results.values()])
        
        print("\n" + "-"*40)
        print(f"平均 MAE: {avg_mae:.4f}")
        print(f"平均 RMSE: {avg_rmse:.4f}")
        print(f"平均 R²: {avg_r2:.4f}")
    
    return results


def save_results(data_splits, eval_results, file_ids, save_dir="results"):
    """保存训练和评估结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'file_ids': file_ids[:50] if len(file_ids) <= 50 else file_ids[:50] + ['...'],
        'total_samples': len(file_ids),
        'train_samples': len(data_splits['train'][0]),
        'val_samples': len(data_splits['val'][0]) if len(data_splits['val']) > 0 else 0,
        'test_samples': len(data_splits['test'][0]),
        'evaluation': eval_results
    }
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    results_path = save_path / f"results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_path}")


def predict_example(factory, processor, agg_method='mean'):
    """预测示例"""
    print("\n" + "="*60)
    print("预测示例")
    print("="*60)
    
    feature_files = processor.find_feature_files()
    if not feature_files:
        print("无可用数据文件")
        return
    
    for feat_path in feature_files[:3]:
        file_id = processor.extract_sample_id(feat_path)
        
        df = processor.load_features_from_csv(feat_path)
        features = processor.aggregate_window_features(df, agg_method=agg_method)
        features = features.reshape(1, -1)
        
        predictions = factory.predict_all(features)
        single_pred = {k: float(v[0]) for k, v in predictions.items()}
        
        K, deviations = factory.calculate_K_value(single_pred)
        
        print(f"\n文件 ID: {file_id}")
        print(f"  整合系数 K: {K:.2f} -> {factory.interpret_K(K)}")


def main():
    parser = argparse.ArgumentParser(description='前庭振动情绪识别系统训练')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_PATH,
                       help=f'数据目录（默认: {DEFAULT_DATA_PATH}）')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--agg_method', type=str, default='all',
                       choices=['mean', 'median', 'std', 'max', 'min', 'all'],
                       help='特征聚合方法')
    parser.add_argument('--epochs', type=int, default=150,
                       help='训练轮数')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--skip_eval', action='store_true',
                       help='跳过评估')
    parser.add_argument('--predict', action='store_true',
                       help='训练后进行预测示例')
    
    args = parser.parse_args()
    
    print("="*60)
    print("前庭振动情绪识别系统 - 训练")
    print("="*60)
    
    # 1. 加载数据
    print(f"\n正在从 {args.data_dir} 加载数据...")
    processor = DataProcessor(data_dir=args.data_dir)
    
    try:
        features, labels, file_ids = processor.prepare_data(
            agg_method=args.agg_method
        )
    except ValueError as e:
        print(f"\n错误: {e}")
        return
    
    # 打印数据摘要
    print_data_summary(features, labels, file_ids)
    
    # 2. 划分数据
    print("\n" + "="*60)
    data_splits = processor.split_data(
        features, labels, 
        test_size=args.test_size, 
        val_size=args.val_size
    )
    
    # 3. 训练模型
    factory, train_results = train_models(
        data_splits, 
        save_dir=args.model_dir,
        epochs=args.epochs
    )
    
    # 4. 评估模型
    eval_results = None
    if not args.skip_eval and len(data_splits['test'][0]) > 0:
        X_test, y_test = data_splits['test']
        eval_results = evaluate_models(factory, X_test, y_test)
        save_results(data_splits, eval_results, file_ids)
    
    # 5. 预测示例
    if args.predict:
        predict_example(factory, processor, agg_method=args.agg_method)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


if __name__ == '__main__':
    main()