import torch
import argparse
import torch.optim as optim
from typing import Iterable
from data.dataset import dataloader
from utils.engine import train_model
from models.hydranet import HydraNet
from losses.uncertainty_loss import MultiTaskLoss
from metrics.metrics import MTLMetrics

def decode_enabled_task(enabled_task_code: str) -> Iterable:
    enabled_task = [False, False, False]
    if "A" in enabled_task_code:
        enabled_task[0] = True
    if "G" in enabled_task_code:
        enabled_task[1] = True
    if "R" in enabled_task_code:
        enabled_task[2] = True
    return enabled_task

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HydraNet()
    
    # parse args
    epochs = args.epochs
    experiment_name = args.experiment_name
    model_dir = args.model_dir
    loss_type = args.loss_type
    enabled_task = decode_enabled_task(enabled_task_code=args.enabled_task_code)
    regression_metric = args.regression_metric
    classification_metric = args.classification_metric

    params = {
        "epochs": epochs,
        "loss_type": loss_type,
        "task_code": args.enabled_task_code,
    }

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = MultiTaskLoss(loss_type=loss_type, task_num=3, enabled_task=enabled_task)
    mtl_metric = MTLMetrics(enabled_task=enabled_task, regression_metric=regression_metric, classification_metric=classification_metric)
    train_loader, val_loader = dataloader(data_dir="data/UTKFace")
    loaders = {"train": train_loader, "val": val_loader}
    print()
    print("="*50)
    train_model(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        loss_fn=loss_fn,
        mtl_metric=mtl_metric,
        device=device,
        model_dir=model_dir,
        params=params,
        experiment_name=experiment_name,
        epochs=epochs,
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
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["learned", "fixed"],
        default="learned",
        metavar="LT",
        help="Type of Loss. Choice: [learned, fixed]. Default: learned"
    )
    parser.add_argument(
        "--enabled-task-code",
        type=str,
        choices=["A", "G", "R", "AG", "AR", "GR", "AGR"],
        default="AGR",
        metavar="ET",
        help="Task that you want to train. A: Age, G: Gender, R: Race. Example: AGR means that you want to include Age, Gender, and Race in training process."
    )
    parser.add_argument(
        "--regression-metric",
        type=str,
        choices=["mae", "mse"],
        default="mae",
        metavar="RM",
        help="Regression metric used to evaluate age regression task."
    )
    parser.add_argument(
        "--classification-metric",
        type=str,
        choices=["f1", "acc", "recall", "precision"],
        default="f1",
        metavar="CM",
        help="Classification metric used to evaluate gender / race classification task."
    )
    args = parser.parse_args()
    main(args)