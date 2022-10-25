import os
import time
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from timeit import default_timer as timer 
from .metrics import MTLMetrics
from typing import Tuple
from tqdm.auto import tqdm

# MLFlow
import mlflow
import mlflow.pytorch as mp

def train_one_epoch(model, 
                    loader, 
                    optimizer,
                    loss_fns: dict) -> Tuple:
    model.train()
    device = next(model.parameters()).device
    batch_losses = []

    mtl_metric = MTLMetrics()
    for imgs, targets in loader:
        imgs = imgs.to(device)
        age_targets = targets['age'].to(device)
        gender_targets = targets['gender'].to(device)
        race_targets = targets['race'].to(device)

        # Batch Result
        optimizer.zero_grad()
        outputs = model(imgs)
        loss_age = loss_fns['age'](outputs[0], age_targets.unsqueeze(1).float())
        loss_gender = loss_fns['gender'](outputs[1], gender_targets.unsqueeze(1).float())
        loss_race = loss_fns['race'](outputs[2], race_targets)
        loss = loss_age + loss_gender + loss_race
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

        mtl_metric.insert(values=age_targets.cpu().numpy(), task="age", mode='target')
        mtl_metric.insert(values=gender_targets.cpu().numpy(), task="gender", mode='target')
        mtl_metric.insert(values=race_targets.cpu().numpy(), task="race", mode='target')
        mtl_metric.insert(values=outputs[0].squeeze(1).detach().numpy(), task="age", mode='output')
        mtl_metric.insert(values=outputs[1].squeeze(1).detach().numpy(), task="gender", mode='output')
        mtl_metric.insert(values=outputs[2].squeeze(1).detach().numpy(), task="race", mode='output')

    

    loss_value = np.mean(batch_losses)
    task_metrics = mtl_metric.evalute_model(regression_metric="mae", classification_metric="f1", step="train")
    metrics = {
        "train_loss_age": loss_age.item(),
        "train_loss_gender": loss_gender.item(),
        "train_loss_race": loss_race.item(),
        "train_total_loss": loss_value.item()
    }
    metrics.update(task_metrics)
    return metrics

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fns: dict):
    model.eval()
    device = next(model.parameters()).device
    batch_losses = []
    mtl_metric = MTLMetrics()

    for imgs, targets in loader:
        imgs = imgs.to(device)
        age_targets = targets['age'].to(device)
        gender_targets = targets['gender'].to(device)
        race_targets = targets['race'].to(device)

        

def train_model(model,
                loader,
                optimizer,
                loss_fns,
                device,
                model_dir: str,
                model_name: str = "Model",
                experiment_name: str = 'Experiment 1',
                epochs: int = 10,
                verbose: bool = True):
    model.to(device)

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id = current_experiment['experiment_id']
    
    with mlflow.start_run(experiment_id=experiment_id):
        start_time = timer()
        for epoch in tqdm(epochs + 1):
            losses = train_one_epoch(model=model, loader=loader, 
                                    optimizer=optimizer, loss_fns=loss_fns)
            mlflow.log_metrics(losses)
            
            if verbose:
                print(f"Epoch: {epoch:3} | Total Loss: {losses['total_loss']:.3} | Loss Age: {losses['loss_age']:.3f} | Loss Gender: {losses['loss_gender']:.3f} | Loss Race: {losses['loss_race']:.3f}")
            
            if epoch == 1:
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=False)
            
            torch.save(model.state_dict(), f"{model_dir}/weight_epoch_{epoch}.pt")
        end_time = timer()
        eval_one_epoch()
        mp.log_model(model, model_name)
        mlflow.log_metrics({"time": end_time - start_time})
        del model
        mlflow.end_run()