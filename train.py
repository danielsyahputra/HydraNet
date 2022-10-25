import torch
import torch.nn as nn
import torch.optim as optim
from utils.engine import train_model
from models.hydranet import HydraNet
from data.dataset import dataloader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HydraNet()
    loss_fns = {
        "age": nn.L1Loss(),
        "gender": nn.BCELoss(),
        "race": nn.CrossEntropyLoss()
    }
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader, val_loader = dataloader(data_dir="data/UTKFace")
    train_model(
        model=model,
        loaders={"train": train_loader, "val": val_loader},
        optimizer=optimizer,
        loss_fns=loss_fns,
        device=device,
        model_dir="output/exp1",
        experiment_name="Uniform Weight Experiment",
        model_name="Model",
        epochs=10
    )

if __name__=="__main__":
    main()