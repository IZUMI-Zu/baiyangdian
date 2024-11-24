from pathlib import Path
from typing import Dict, Tuple

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import seaborn as sns


class Evaluator:
    def __init__(self, model, device, save_dir: Path):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, test_loader) -> Dict[str, float]:
        """
        评估模型性能并返回多个评估指标
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算各种评估指标
        metrics = {
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions)
        }
        
        # 生成评估图表
        self._plot_predictions(predictions, targets)
        self._plot_residuals(predictions, targets)
        
        return metrics
    
    def _plot_predictions(self, predictions: np.ndarray, targets: np.ndarray):
        """
        绘制预测值vs真实值的散点图
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Prediction vs True Values')
        plt.savefig(self.save_dir / 'predictions.png')
        plt.close()
    
    def _plot_residuals(self, predictions: np.ndarray, targets: np.ndarray):
        """
        绘制残差分布图
        """
        residuals = predictions - targets
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.ylabel('Count')
        plt.title('Residuals Distribution')
        plt.savefig(self.save_dir / 'residuals.png')
        plt.close()

    @staticmethod
    def load_best_model(model, checkpoint_path: Path) -> Tuple[torch.nn.Module, dict]:
        """
        加载最佳模型
        """
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint