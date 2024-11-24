import torch
import logging
from pathlib import Path
from tqdm import tqdm

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
        
        self.best_val_loss = float('inf')
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch_features, batch_labels in pbar:
                # 训练步骤实现
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    epoch,
                    is_best=True,
                    val_loss=val_loss
                )
    
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