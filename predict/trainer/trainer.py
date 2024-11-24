import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import time

class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        save_dir,
        scheduler=None,
        logger=None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.scheduler = scheduler
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化 TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.save_dir / 'tensorboard')
        self.best_val_loss = float('inf')
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, (batch_features, batch_labels) in enumerate(pbar):
                # 训练步骤实现
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                # 记录每个批次的损失
                total_loss += loss.item()
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Loss/train_step', loss.item(), step)
                
                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/(batch_idx+1):.4f}"
                })
                
                # 可选：记录梯度和权重分布
                if batch_idx % 100 == 0:
                    for name, param in self.model.named_parameters():
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, step)
                        self.writer.add_histogram(f'Weights/{name}', param.data, step)
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                total_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_labels.cpu().numpy())
        
        # 记录验证集性能指标
        avg_val_loss = total_loss / len(val_loader)
        self.writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        return avg_val_loss
    
    def train(self, train_loader, val_loader, num_epochs):
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, epoch)
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            
            # 验证
            val_loss = self.validate(val_loader, epoch)
            
            # 学习率调整
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                self.writer.add_scalar(
                    'Learning_rate',
                    self.optimizer.param_groups[0]['lr'],
                    epoch
                )
            
            # 记录每个epoch的时间
            epoch_time = time.time() - epoch_start_time
            self.writer.add_scalar('Time/epoch', epoch_time, epoch)
            
            # 记录日志
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    epoch,
                    is_best=True,
                    val_loss=val_loss
                )
        
        # 记录总训练时间
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes")
        self.writer.close()
    
    def save_checkpoint(self, epoch, is_best=False, **kwargs):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            **kwargs
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")