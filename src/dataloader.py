"""
This module corresponds to the dataloader. It contains the function generating the examples with noise, as well as,
the Dataset as generator, and finally a function get_datasets which creates the three dataloaders from the split
"""
# Python modules
from typing import Optional, Tuple, Generator
# ML general
import numpy as np
# DL
import torch
from torch.utils.data import Dataset, DataLoader
# Local modules
from src.tools import CircleParams, noisy_circle


def generate_examples(
        noise_level: float = 0.5,
        img_size: int = 100,
        min_radius: Optional[int] = None,
        max_radius: Optional[int] = None,
        dataset_path: str = 'ds',
) -> Generator[Tuple[np.ndarray, CircleParams], None, None]:
    """
    This class generates an example of circle with noise. It outputs the image as an array on gray scale, and the
    parameters of the circle as tool.CircleParams class
    """
    if not min_radius:
        min_radius = img_size // 10
    if not max_radius:
        max_radius = img_size // 2
    assert max_radius > min_radius, "max_radius must be greater than min_radius"
    assert img_size > max_radius, "size should be greater than max_radius"
    assert noise_level >= 0, "noise should be non-negative"

    params = f"{noise_level=}, {img_size=}, {min_radius=}, {max_radius=}, {dataset_path=}"
    print(f"Using parameters: {params}")
    while True:
        img, params = noisy_circle(
            img_size=img_size, min_radius=min_radius, max_radius=max_radius, noise_level=noise_level
        )
        yield img, params


class CircleDataset(Dataset):
    """
    This Dataset class generates examples of a generator from noisy_circle under the given hyperparameters from Config
    class. It is designed to be combined with a torch.utils.data.DataLoader for each one of the splits: Train, Val, Test
    """
    def __init__(self, config , mode: str = 'train') -> None:
        self.config = config
        self.mode = mode
        self.num = config.data_config.num_train if mode == 'train' \
                            else config.data_config.num_val if mode == 'val' \
                            else config.data_config.num_test
        self.data = self.generator()

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, params = next(self.data)
        img = img.unsqueeze(0)
        return img, params

    def generator(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """ This is the generator based on noisy_circle and converted into float32"""
        while True:
            # Create example
            img, params = noisy_circle(
                self.config.circle.img_size,
                self.config.circle.min_radius,
                self.config.circle.max_radius,
                self.config.circle.noise_level)
            # Datatype conversion
            params = torch.tensor(params, dtype=torch.float32, device=self.config.device)
            img = torch.tensor(img, dtype=torch.float32, device=self.config.device)
            yield img, params

def get_datasets(config: dict)-> tuple[DataLoader, DataLoader, DataLoader]:
    """ Get dataloaders for train, val and test sets. """
    train_loader = DataLoader(CircleDataset(config, mode='train'), batch_size=config.data_config.batch_size)
    val_loader = DataLoader(CircleDataset(config, mode='val'), batch_size=config.data_config.batch_size)
    test_loader = DataLoader(CircleDataset(config, mode='test'), batch_size=config.data_config.batch_size)
    return train_loader, val_loader, test_loader
