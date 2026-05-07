import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cross_entropy import CustomEntropyLoss
import pytorch_check
from constants import (
    EPOCH_COUNT,
    LEARNING_RATE,
    SAVE_ON_EPOCH,
    DISPLAY_STEP_SIZE
)
from llm import model
from parse_dataset import StoryDataset

pytorch_check.display_accelerator()


def train():
    total_parameters = sum(p.numel() for p in model.parameters())
    print("This model has: ", total_parameters, " parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CustomEntropyLoss()

    dataset = StoryDataset()
    loaded_dataset = DataLoader(dataset, batch_size=1, shuffle=False)
    total_steps = len(loaded_dataset)

    for epoch in range(1, EPOCH_COUNT + 1):
        start_time = time.perf_counter()
        total_loss = 0
        for i, (inputs, targets) in enumerate(loaded_dataset):
            logits: torch.Tensor = model(inputs[0])

            # compare the output from the model given the first token (input[0]) to the given story next token (targets[0])
            loss = loss_fn(logits, targets[0])
            total_loss += loss.item()

            optimizer.zero_grad()  # reset parameter gradients across model
            loss.backward()  # calculate gradients using loss
            optimizer.step()  # updates parameters through the whole model

            if i % DISPLAY_STEP_SIZE == 0:
                print(f"Epoch {epoch} | Step {i}/{total_steps} | Loss: {loss.item()}")

        if epoch % SAVE_ON_EPOCH == 0:
            torch.save(model, f"dist/model-{pytorch_check.device}-{epoch}.pth")
            print(f"Saved snapshot at: dist/model-{pytorch_check.device}-{epoch}.pth")

        average_loss = total_loss / total_steps
        end_time = time.perf_counter()

        time_taken = end_time - start_time
        eta_seconds = (EPOCH_COUNT - epoch) * time_taken
        time_until_completion = datetime.timedelta(seconds=int(eta_seconds))

        print(
            f"Epoch {epoch} | Loss: {average_loss} | {time_taken:.4f} seconds/epoch | ETA: {str(time_until_completion)}"
        )

    print()  # new line at the end
    torch.save(model, f"dist/model-{pytorch_check.device}.pth")


train()
