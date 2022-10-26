import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from data.dataset import dataloader
from utils.engine import train_model
from models.hydranet import HydraNet

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HydraNet()
    loss_fns = {
        "age": nn.L1Loss(),
        "gender": nn.BCELoss(),
        "race": nn.CrossEntropyLoss()
    }
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader, val_loader = dataloader(data_dir="data/UTKFace")
    epochs = args.epochs
    experiment_name = args.experiment_name
    model_dir = args.model_dir
    print("="*200)
    train_model(
        model=model,
        loaders={"train": train_loader, "val": val_loader},
        optimizer=optimizer,
        loss_fns=loss_fns,
        device=device,
        model_dir=model_dir,
        experiment_name=experiment_name,
        model_name="Model",
        epochs=epochs
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="HydraNet Using Uncertainty to Weigh Losses")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of epochs for training (default: 10)."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Experiment All",
        metavar="EN",
        help="Name of Experiment in MLFLow."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="exp1",
        metavar="MD",
        help="Name of directory for saving the model and artifacts."
    )
    args = parser.parse_args()
    main(args)