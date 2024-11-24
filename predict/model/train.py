import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import argparse

from ..data_processing import BaiyangdianDataset
from .network import WaterLevelCNN

logger = logging.getLogger(__name__)

def train_model(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize dataset and data loader
    dataset = BaiyangdianDataset(
        data_dir=args.data_dir,
        target_column=args.target_column,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = WaterLevelCNN().to(device)
    criterion = nn.MSELoss()  # Assuming regression for water level prediction
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Resume training from a checkpoint if specified
    start_epoch = 0

    if args.checkpoint and Path(args.checkpoint).is_file():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed training from checkpoint: {args.checkpoint}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0

        for batch_features, batch_labels in dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            predictions = model(batch_features)
            loss = criterion(predictions.squeeze(), batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = Path(args.save_dir) / f"model_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            logger.info(f"Checkpoint saved at: {checkpoint_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the WaterLevelCNN model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data directory.")
    parser.add_argument('--target_column', type=str, required=True, help="Target column name.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to resume checkpoint.")
    parser.add_argument('--checkpoint_interval', type=int, default=5, help="Save checkpoint every N epochs.")
    parser.add_argument('--save_dir', type=str, default="./checkpoints", help="Directory to save checkpoints.")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    train_model(args)
