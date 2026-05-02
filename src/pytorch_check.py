import torch

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def display_accelerator():
    print("PyTorch Accelerator: ", device)


if __name__ == "__main__":
    display_accelerator()
