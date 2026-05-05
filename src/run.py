import torch

from constants import MAX_LENGTH
from parse_dataset import index_to_token_map, string_to_token_ids
import pytorch_check


def run_forward_pass(input):
    token_ids = string_to_token_ids(input)

    if len(token_ids) > MAX_LENGTH:
        raise ValueError("Input exceeds MAX_LENGTH of ", MAX_LENGTH, " tokens.")

    output: torch.Tensor = model(token_ids)

    next_token_index = int(output[-1].argmax().item())
    return index_to_token_map[next_token_index]


def generate(prompt: str, new_tokens=30):
    current_string = prompt

    for _ in range(new_tokens):
        new_token = run_forward_pass(current_string)

        separator = " "
        if new_token in ("!", ".", "?", "'", ",") or current_string[-1] == "'":
            separator = ""

        current_string = current_string + separator + new_token

    print("Output: ", current_string)


model = torch.load(f"dist/model-{pytorch_check.device}-5.pth", weights_only=False)

total_parameters = sum(p.numel() for p in model.parameters())
print("This model has: ", total_parameters, " parameters.")

input_string = input("Enter a prompt (or press Enter for default): ") or "one day,"
token_length = int(input("Enter the number of tokens you wish to generate: ")) or 30
generate(input_string, token_length)
