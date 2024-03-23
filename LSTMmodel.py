import numpy as np
import pandas as pd
import random

import os
import torch
import json
import string
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
from collections import OrderedDict

from tldextract import extract

from tqdm import tqdm
from datetime import datetime
import time
from model_api.src.ml_api import *
import matplotlib.pyplot as plt

NUM_ROUND = 2
ROUND_DICT = {}
batch_size = 64
lr = 3e-4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(max_features, embed_size, hidden_size, n_layers).to(device)

my_df = save_dataframe(sys.argv[1])
trainloader, testloader = split_train_test_data(my_df)

def do_evaluate_round():
    for round_idx in range(NUM_ROUND):
        print(f"\nEvaluate Round {round_idx + 1}:\n")
        model_path = f"./model_round_{round_idx + 1}.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))

        criterion = nn.BCELoss(reduction='mean')
        optimizer = optim.RMSprop(params=model.parameters(), lr=lr)

        eval_loss, accuracy = test(model=model, testloader=testloader, criterion=criterion, batch_size=batch_size)
        print(
            "Eval Loss: {:.4f}".format(eval_loss),
            "Accuracy: {:.4f}".format(accuracy)
        )

        ROUND_DICT[f"round_{round_idx + 1}"] = {
            "accuracy": accuracy,
            "eval_loss": eval_loss
        }
    print(ROUND_DICT)


#start_training_task()
if __name__ == "__main__":
    do_evaluate_round()
    # Extract accuracy values from round_dict
    accuracies = [ROUND_DICT[f"round_{i+1}"]["accuracy"] for i in range(NUM_ROUND)]

    # Extract accuracy and avg_loss values from round_dict
    accuracies = [ROUND_DICT[f"round_{i+1}"]["accuracy"] for i in range(NUM_ROUND)]
    avg_losses = [ROUND_DICT[f"round_{i+1}"]["eval_loss"] for i in range(NUM_ROUND)]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axs[0].plot(range(1, NUM_ROUND + 1), accuracies, marker='o')
    axs[0].set_title('Accuracy over rounds')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].set_xticks(range(1, NUM_ROUND + 1))
    axs[0].grid(True)

    # Plot average loss
    axs[1].plot(range(1, NUM_ROUND + 1), avg_losses, marker='o', color='red')
    axs[1].set_title('Average Loss over rounds')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Average Loss')
    axs[1].set_xticks(range(1, NUM_ROUND + 1))
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()