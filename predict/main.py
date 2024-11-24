from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging


from .evaluate.evaluate import Evaluator
from .data_processing import BaiyangdianDataset, create_data_loaders
from .model.network import WaterLevelCNN
from .trainer.trainer import Trainer
from .utils.logger import setup_logger

def main(args):
    # 设置日志
    setup_logger(args.log_dir)
    logger = logging.getLogger(__name__)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载数据集
    dataset = BaiyangdianDataset(
        data_dir=args.data_dir,
        target_column=args.target_column
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # 初始化模型
    model = WaterLevelCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        scheduler=scheduler,
        logger=logger
    )
    
    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs
    )

    evaluator = Evaluator(
        model=model,
        device=device,
        save_dir=args.save_dir / 'evaluation'
    )

    best_model_path = Path(args.save_dir) / 'best_model.pth'
    model, _ = Evaluator.load_best_model(model, best_model_path)
    metrics = evaluator.evaluate(test_loader)
    
    logger.info("Test Set Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--target_column', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    
    args = parser.parse_args()
    main(args)