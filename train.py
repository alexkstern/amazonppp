from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import numpy as np

from model import PPP
from dataset import PPPDataset

# HPARAMS
dim = 768
num_blocks = 12
heads = 8
mlp_hidden_dim = 2048
pad_length = 16

num_training_steps = 1000
learning_rate = 3e-4

val_size = 0.10
batch_size = 32

save_every = 250
val_every = 250
seed = 42


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Create model
    model = PPP(
        dim=dim,
        num_blocks=num_blocks,
        heads=heads,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    # print num parameters in Millions
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of parameters: {num_params:.2f}M")

    # Load dataset
    dataset = PPPDataset(
        pad_length=pad_length,
    )
    split = int(len(dataset) * val_size)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - split, split])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # print len of train and val
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(train_loader)}")

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Move model and loss function to device
    model.to(device)

    # Train the model
    pbar = tqdm(range(int(num_training_steps)))
    for i in pbar:
        batch: dict = next(iter(train_loader))
        batch = {k: v.to(device) for k, v in batch.items()}  # Send batch to device
        optimizer.zero_grad()
        loss = model.loss(**batch)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")

        if i % save_every == 0:
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'./checkpoints/{i}.pt')

        if i % val_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in val_loader:
                    val_loss += model.loss(**batch)
                val_loss /= len(val_loader)
                pbar.set_postfix(val_loss=f"{val_loss.item():.4f}")
            model.train()
    

if __name__ == '__main__':
    main()
