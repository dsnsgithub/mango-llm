import datetime
import time

import torch
import torch.nn as nn

import pytorch_check
from constants import (
    EPOCH_COUNT,
    LEARNING_RATE,
    SAVE_ON_EPOCH,
)
from llm import model
from parse_dataset import dataset, string_to_token_ids

pytorch_check.display_accelerator()


def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    tokenized_stories = []
    for story in dataset:
        token_ids = string_to_token_ids(story)
        tokenized_stories.append(token_ids)

    for epoch in range(EPOCH_COUNT):
        start_time = time.perf_counter()
        total_loss = 0
        for story_tokens in tokenized_stories:
            inputs = story_tokens[:-1]  # everything except last token
            targets = story_tokens[1:]  # everything after first token

            logits: torch.Tensor = model(inputs)

            # compare the output from the model given the first token (input[0]) to the given story next token (targets[0])
            loss = loss_fn(logits, targets)
            total_loss += loss.item()

            optimizer.zero_grad()  # reset parameter gradients across model
            loss.backward()  # calculate gradients using loss
            optimizer.step()  # updates parameters through the whole model

        if epoch % SAVE_ON_EPOCH == 0 and epoch != 0:
            torch.save(
                model.state_dict(), f"dist/model-{pytorch_check.device}-{epoch}.pth"
            )
            print(f"Saved snapshot at: dist/model-{pytorch_check.device}-{epoch}.pth")

        average_loss = total_loss / len(tokenized_stories)
        end_time = time.perf_counter()

        time_taken = end_time - start_time
        eta_seconds = (EPOCH_COUNT - epoch) * time_taken
        time_until_completion = datetime.timedelta(seconds=int(eta_seconds))

        print(
            f"Epoch {epoch} | Loss: {average_loss} | Time Taken: {time_taken} sec | ETA: {str(time_until_completion)}"
        )

    print()  # new line at the end
    torch.save(model.state_dict(), f"dist/model-{pytorch_check.device}.pth")


train()


total_parameters = sum(p.numel() for p in model.parameters())
print("This model has: ", total_parameters, " parameters.")
