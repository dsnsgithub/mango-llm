import torch

from constants import RANDOM_SEED

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

torch.manual_seed(RANDOM_SEED)


def display_accelerator():
    print("PyTorch Accelerator: ", device)


if __name__ == "__main__":
    display_accelerator()
