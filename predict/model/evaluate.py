import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import logging

from ..data_processing import BaiyangdianDataset
from .network import WaterLevelCNN

logger = logging.getLogger(__name__)

def evaluate_model(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = BaiyangdianDataset(
        data_dir=args.data_dir,
        target_column=args.target_column,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = WaterLevelCNN().to(device)

    # Load checkpoint
    if not Path(args.checkpoint).is_file():
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded from checkpoint: {args.checkpoint}")

    # Evaluation mode
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()  # Assuming regression for water level prediction

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Predictions
            predictions = model(batch_features)
            loss = criterion(predictions.squeeze(), batch_labels)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    logger.info(f"Evaluation complete. Average Loss: {average_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the WaterLevelCNN model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data directory.")
    parser.add_argument('--target_column', type=str, required=True, help="Target column name.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    args = parser.parse_args()

    evaluate_model(args)
