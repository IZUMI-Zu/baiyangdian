import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Tuple, Callable, Optional
import logging


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

    def _load_data(self) -> None:
        """Load raw data from all files in the data directory."""
        all_data = []

        try:
            # Traverse sub folders and load data files
            for folder in self.data_dir.iterdir():
                if folder.is_dir():
                    for file in folder.iterdir():
                        if file.suffix in ['.csv', '.nc']:  # Adjust supported file types as needed
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
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.nc':
            from .load_data import load_nc_data  # Replace with your own NetCDF loader
            return load_nc_data(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.processed_data is not None:
            return len(self.processed_data)
        raise ValueError("No data has been processed.")

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
        return feature, label
