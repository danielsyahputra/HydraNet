# HydraNet
HydraNet Using Uncertainty to Weigh Losses for Face Attribute Recognition

## Retraining

```
git clone https://github.com/danielsyahputra/HydraNet.git
```
```
cd HydraNet/
```
```
python3 download.py
```
```
python3 train.py --epochs <EPOCH> \ 
  --experiment-name <FILL> \ 
  --model-dir <FILL> \ 
  --loss-type <FILL> \ 
  --enabled-task-code <FILL> \
  --regression-metric <FILL> \ 
  --classification-metric <FILL>
```

Args:
```
--epochs : Number of epochs for training (default: 10).
--experiment-name : Name of Experiment in MLFLow
--model-dir : Name of directory for saving the model and artifacts.
--loss-type : Type of Loss. If you choose learned, then the weight for each task will be got from training process. Choice: [learned, fixed]. Default: learned
--enabled-task-code : Task that you want to train. A: Age, G: Gender, R: Race. Example: AGR means that you want to include Age, Gender, and Race in training process. Choices: [A, G, R, AG, AR, ..., AGR]
--regression-metric : Regression metric used to evaluate age regression task.
--classification-metric: Classification metric used to evaluate gender / race classification task.
```
