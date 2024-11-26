import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, Tuple, Callable, Optional
import logging

from . import load_data

class BaiyangdianDataset(Dataset):
    """
    A flexible dataset class for handling Baiyangdian water level prediction data.
    Designed to load data from multiple files and folders while leaving space for preprocessing.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        target_column: str,
        transform: Optional[Callable] = None,
        preprocess: Optional[Callable] = None,
    ):
        """
        Args:
            data_dir (str | Path): Root directory containing sub folders with data files.
            target_column (str): Name of the column in the data representing the target (e.g., water level).
            transform (Callable, optional): Transformation function for features during __getitem__. Default is None.
            preprocess (Callable, optional): Preprocessing function applied to raw data after loading. Default is None.
        """
        self.data_dir = Path(data_dir)
        self.target_column = target_column
        self.transform = transform
        self.preprocess = preprocess
        self.logger = logging.getLogger(__name__)

        # Containers for raw and processed data
        self.raw_data = None  # Raw merged data as a DataFrame
        self.processed_data = None  # Processed data ready for training (features and labels)

        # Load raw data
        self._load_data()

        # Add properties to store feature dimensions
        self.input_height = None
        self.input_width = None
        self._calculate_input_dimensions()

    def _load_data(self) -> None:
        """Load raw data from all files in the data directory."""
        all_data = []

        try:
            # Traverse sub folders and load data files
            for folder in self.data_dir.iterdir():
                if folder.is_dir():
                    for file in folder.iterdir():
                        if file.suffix in ['.csv', '.nc', '.xls', '.xlsx']:
                            data = self._load_file(file)
                            all_data.append(data)

            # Merge all loaded data into a single DataFrame
            self.raw_data = pd.concat(all_data, ignore_index=True)

            # Apply preprocessing if defined
            if self.preprocess:
                self.processed_data = self.preprocess(self.raw_data)
            else:
                self.processed_data = self.raw_data

            self.logger.info("Data successfully loaded and processed.")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load an individual file."""
        return load_data(file_path)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.processed_data is not None:
            return len(self.processed_data)
        raise ValueError("No data has been processed.")

    def _calculate_input_dimensions(self):
        """Calculate the input dimensions from the processed data."""
        if self.processed_data is None:
            raise ValueError("No data has been processed.")
            
        # Debug: Print available columns
        self.logger.info(f"Available columns: {self.processed_data.columns.tolist()}")
        
        # Get the shape of the first feature
        first_feature = self.processed_data.iloc[0].drop(self.target_column).values
        # Assuming the feature can be reshaped into a square matrix
        # Exclude the channel dimension (63)
        feature_size = len(first_feature) // 63
        self.input_height = int(feature_size ** 0.5)
        self.input_width = self.input_height
        
        self.logger.info(f"Input dimensions calculated: {self.input_height}x{self.input_width}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Features and target as tensors.
        """
        if self.processed_data is None:
            raise ValueError("No data has been processed.")

        # Extract a single row of data
        row = self.processed_data.iloc[idx]
        feature = row.drop(self.target_column).values
        label = row[self.target_column]

        # Apply optional transformations
        if self.transform:
            feature = self.transform(feature)

        # Convert to PyTorch tensors
        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Reshape the feature tensor to (channels, height, width)
        feature = feature.reshape(63, self.input_height, self.input_width)
        return feature, label


def create_data_loaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """
    使用scikit-learn划分数据集并创建DataLoader
    """
    # 计算数据集大小
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # 使用scikit-learn进行划分
    train_indices, temp_indices = train_test_split(
        range(total_size), 
        train_size=train_size,
        random_state=random_state
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=random_state
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices)
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices)
    )
    
    return train_loader, val_loader, test_loader