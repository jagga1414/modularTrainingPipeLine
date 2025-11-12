import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from dataset.dataset import MyDataset, MyCollate, Batch_D
from model.model_loader import setup_tokenizer, load_model
from train.trainer import train


# Setup tokenizer and model
tokenizer = setup_tokenizer()
model = load_model()

# Create dataset and dataloader
mydataset = MyDataset("Customer-Support.csv", tokenizer)
mycollate = MyCollate(tokenizer)

# Setup config
config = Batch_D()
config.max_epoch = 4
config.batch_size = 4
config.device = "cpu"

# Create dataloader
mydataloader = DataLoader(mydataset, batch_size=config.batch_size, shuffle=True, collate_fn=mycollate)

# Train
model.to(config.device)
train(model, mydataloader, config)

# Save model
torch.save(model.state_dict(), "model.pt")
print("Training complete! Model saved to model.pt")
