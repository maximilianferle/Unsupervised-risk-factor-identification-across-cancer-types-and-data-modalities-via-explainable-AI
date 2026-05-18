from abc import abstractmethod
from os.path import isfile
from warnings import warn

import torch
from lightning import LightningDataModule
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.csv_preprocessing import get_feat_surv_df
from data.save_load import load, save
from utils import ROOT


class SurvivalDataset(Dataset):
    """A PyTorch Dataset for survival data.

    This dataset stores features, survival durations, and event observations (i.e. censorship / event occurrence)
    and returns them as tuples for training survival models.

    Attributes:
        features (torch.Tensor): Input features for the dataset.
        durations (torch.Tensor): Survival durations for each sample.
        event_observed (torch.Tensor): Binary event observations (1 if event occurred, 0 if censored).
    """
    def __init__(self, features, durations, event_observed):
        self.features = features
        self.durations = durations
        self.event_observed = event_observed

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a single sample or slices from the dataset.

        Args:
            idx (int): Index of samples to retrieve.

        Returns:
            tuple: A tuple containing:
                - features (torch.Tensor): Input features for the sample.
                - tuple: (durations, event_observed) for the sample.
        """
        return self.features[idx], (self.durations[idx], self.event_observed[idx])


class BaseDataModule(LightningDataModule):
    """Base PyTorch Lightning DataModule for survival datasets.

    This class provides core functionality for loading, splitting, and preprocessing survival data.
    It supports caching, train-validation splits, and standardization of features.

    Args:
        seed (int): Random seed for reproducibility. Default: 0.
        batch_size (int): Batch size for DataLoaders. Default: 32.
        train_val_split (float): Ratio of training data to total data. Default: 0.8.
        k (int): Fold number for cross-validation splits. Default: 1.
        use_cached (bool): Whether to load cached data if available. Default: False.
    """
    def __init__(self,
                 seed: int = 0,
                 batch_size: int = 32,
                 train_val_split: float = 0.8,
                 k: int = 1,
                 use_cached: bool = False,
                 ):
        super(BaseDataModule, self).__init__()
        self.seed = seed
        self.k = k

        if not (use_cached and self._load_if_cached()):
            self._init(batch_size=batch_size, train_val_split=train_val_split)
            self.save()

    def _load_if_cached(self):
        """Loads cached data if available.

        Returns:
            bool: True if cached data was loaded, False otherwise.
        """
        name = self.__class__.__name__
        file_path = ROOT / f"data/datamodules/cache/{name}.{self.seed}.{self.k}.pkl"
        is_cached = isfile(file_path)
        if is_cached:
            warn(f"\n{name} was loaded from cache.\n"
                 f"Attributes `seed` and `k` are equal, but `batch_size` and `train_val_split` may be different.\n"
                 f"If this is not desired, set use_cached=False")
            self.__dict__.update(load(file_path))
        return is_cached

    def save(self):
        """Saves the current state of the DataModule to disk for caching."""
        name = self.__class__.__name__
        file_path = ROOT / f"data/datamodules/cache/{name}.{self.seed}.{self.k}.pkl"
        save(self.__dict__, file_path)

    def _init(
            self,
            batch_size: int = 64,
            train_val_split: float = 0.8,
    ):
        """Initializes the DataModule by loading data and splitting it into train/val/test sets.

        Args:
            batch_size (int): Batch size for DataLoaders. Default: 64.
            train_val_split (float): Ratio of training data to total data. Default: 0.8.
        """
        self.df_feat, self.df_surv = self.load_data()
        self.input_size = self.df_feat.shape[1]
        self.batch_size = batch_size

        self.idx_train, self.idx_val = self._get_split_idxs(split_ratio=train_val_split)
        self._train, self._val, self._test = self._split_data(self.idx_train, self.idx_val)
        self._standard_scaler = self._fit_standard_scaler()

    @abstractmethod
    def load_data(self) -> (DataFrame, DataFrame):
        """Loads and returns feature and survival data as pandas DataFrames.

        Returns:
            tuple: A tuple containing:
                - df_feat (DataFrame): Feature DataFrame.
                - df_surv (DataFrame): Survival DataFrame with 'durations' and 'event_observed' columns.
        """

    def setup(self, stage=None):
        """Sets up datasets for training, validation, and testing to be used in the lightning framework.

        Args:
            stage (str, optional): Stage of training (fit, test, or None). Default: None.
        """
        # Prepare datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = self._make_dataset(self._train)
            self.val_dataset = self._make_dataset(self._val)

        if stage == 'test' or stage is None:
            self.test_dataset = self._make_dataset(self._test)

    def train_dataloader(self):
        """Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        """Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def test_dataloader(self):
        """Returns a DataLoader for the test dataset.

        Returns:
            DataLoader: Test DataLoader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        """Returns a DataLoader for prediction.

        Returns:
            DataLoader: Prediction DataLoader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def _make_dataset(self, data: (torch.tensor, torch.tensor, torch.tensor)) \
            -> SurvivalDataset:
        """Creates a SurvivalDataset from input tensors.

        Args:
            data (tuple): A tuple containing:
                - features (torch.Tensor): Input features.
                - durations (torch.Tensor): Survival durations.
                - event_observed (torch.Tensor): Event observations.

        Returns:
            SurvivalDataset: Dataset for survival analysis.
        """
        features, durations, event_observed = data
        features = self._standard_scaler.transform(features.numpy())
        features = torch.tensor(features, dtype=torch.float32)

        return SurvivalDataset(features, durations, event_observed)

    def _get_split_idxs(self, split_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates train and validation indices for cross-validation.

        Args:
            split_ratio (float): Ratio of validation data to total data.

        Returns:
            tuple: Training and validation indices as tensors.
        """
        torch.random.manual_seed(self.seed)

        assert 0 <= split_ratio <= 1, 'split_ratio must be between 0 and 1'
        split_ratio = round(min(split_ratio, 1 - split_ratio), ndigits=1)
        k = round(min(self.k, 1 / split_ratio))

        n_samples = len(self.df_feat)
        indices = torch.randperm(n_samples)
        n_val_samples = int(n_samples * split_ratio)

        idx_val = indices[(k - 1) * n_val_samples:k * n_val_samples]
        uniques, counts = torch.cat((idx_val, indices)).unique(return_counts=True)
        idx_train = uniques[counts == 1]

        return idx_train, idx_val

    def _split_data(self, idx_train, idx_val) -> tuple[tuple, tuple, tuple]:
        """Splits data into train, validation, and test sets.

        Args:
            idx_train (torch.Tensor): Indices for training data.
            idx_val (torch.Tensor): Indices for validation data.

        Returns:
            tuple: Train, validation, and test data as tuples of tensors.
        """
        features = torch.tensor(self.df_feat.values, dtype=torch.float32)
        durations = torch.tensor(self.df_surv['durations'].values, dtype=torch.float32)
        event_observed = torch.tensor(self.df_surv['event_observed'].values, dtype=torch.float32)

        train_data = tuple(tensor[idx_train] for tensor in (features, durations, event_observed))
        val_data = tuple(tensor[idx_val] for tensor in (features, durations, event_observed))
        test_data = val_data

        return train_data, val_data, test_data

    def _fit_standard_scaler(self) -> StandardScaler:
        """Fits a StandardScaler to the training data.

        Returns:
            StandardScaler: Fitted scaler for standardizing features.
        """
        scaler = StandardScaler()
        scaler.fit(self._train[0])

        return scaler


class CoMMpassDataModule(BaseDataModule):
    """DataModule for the CoMMpass dataset.

    This class loads and preprocesses the CoMMpass dataset for survival analysis.
    It includes imputation for missing values in the feature data.
    """
    def load_data(self):
        """Loads the CoMMpass dataset with features and survival data.

        Returns:
            tuple: Feature and survival DataFrames.
        """
        from data.csv_preprocessing import get_nan_containing_feat_surv_df
        return get_nan_containing_feat_surv_df(survival="os")

    def _split_data(self, idx_train, idx_val) -> tuple[tuple, tuple, tuple]:
        """Splits data into train, validation, and test sets with imputation for missing values.

        Args:
            idx_train (torch.Tensor): Indices for training data.
            idx_val (torch.Tensor): Indices for validation data.

        Returns:
            tuple: Train, validation, and test data as tuples of tensors.
        """
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        train_data = (
            imputer.fit_transform(self.df_feat.values[idx_train]),
            self.df_surv['durations'].values[idx_train],
            self.df_surv['event_observed'].values[idx_train],
        )
        train_data = tuple(torch.tensor(x, dtype=torch.float32) for x in train_data)

        val_data = (
            imputer.transform(self.df_feat.values[idx_val]),
            self.df_surv['durations'].values[idx_val],
            self.df_surv['event_observed'].values[idx_val],
        )
        val_data = tuple(torch.tensor(x, dtype=torch.float32) for x in val_data)

        test_data = val_data
        return train_data, val_data, test_data


class ImageBaseDataModule(BaseDataModule):
    """Base DataModule for image-based survival datasets.
    """
    def _init(self,
              batch_size: int = 64,
              train_val_split: float = 0.8, ):
        """Initializes the DataModule by loading data and splitting it into train/val/test sets.

        Args:
            batch_size (int): Batch size for DataLoaders. Default: 64.
            train_val_split (float): Ratio of training data to total data. Default: 0.8.
        """
        self.load_data()
        self.df_feat = self.images
        self.input_size = self.images.shape[1]
        self.batch_size = batch_size

        self.idx_train, self.idx_val = self._get_split_idxs(split_ratio=train_val_split)
        self._train, self._val, self._test = self._split_data(self.idx_train, self.idx_val)

    def _split_data(self, idx_train, idx_val) -> tuple[tuple, tuple, tuple]:
        """Splits image data into train, validation, and test sets.

        Args:
            idx_train (torch.Tensor): Indices for training data.
            idx_val (torch.Tensor): Indices for validation data.

        Returns:
            tuple: Train, validation, and test data as tuples of tensors.
        """
        inputs = (self.images, self.survival_times, self.event_observed)
        train_data = tuple(tensor[idx_train] for tensor in inputs)
        val_data = tuple(tensor[idx_val] for tensor in inputs)
        test_data = val_data
        return train_data, val_data, test_data

    def _make_dataset(self, data: (torch.tensor, torch.tensor, torch.tensor)) -> SurvivalDataset:
        """Creates a SurvivalDataset from image data.

        Args:
            data (tuple): A tuple containing:
                - images (torch.Tensor): Image data.
                - survival_times (torch.Tensor): Survival durations.
                - event_observed (torch.Tensor): Event observations.

        Returns:
            SurvivalDataset: Dataset for survival analysis with image data.
        """
        images, survival_times, event_observed = data
        return SurvivalDataset(images[:, None, :, :], survival_times, event_observed)
