# ResNet-18 / ViT-Tiny Pet Classification

This project implements pet classification on the Oxford-IIIT Pet dataset using PyTorch. It supports training and evaluation with both ResNet-18 and ViT-Tiny models.

## Project Structure

- `main.py`: Training entrypoint with model selection and hyperparameter options.
- `dataset.py`: Dataset loading and preprocessing using `torchvision.datasets.OxfordIIITPet`.
- `model.py`: Model definitions for `ResNet18` and `ViTTiny`.
- `trainer.py`: Training and evaluation logic.
- `requirements.txt`: Python package dependencies.

## Dependencies

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

- `torch` / `torchvision` / `torchaudio`
- `timm`
- `numpy`
- `matplotlib`
- `tqdm`
- `swanlab`

## Dataset

This project uses the `OxfordIIITPet` dataset from `torchvision`:

- Training split: `trainval`
- Test split: `test`

The dataset is downloaded automatically into the `./data` folder.

## Usage

Run the main script from the command line:

```bash
python main.py [--model resnet18|vittiny] [--scratch] [--epochs N] [--batch-size B] [--img-size S] [--lr-backbone LR1] [--lr-fc LR2] [--weight-decay WD]
```

### Common arguments

- `--model`: Choose the model type (`resnet18` by default, or `vittiny`).
- `--scratch`: Train from scratch (only available for `resnet18`).
- `--epochs`: Number of training epochs (default: `300`).
- `--batch-size`: Batch size (default: `64`).
- `--img-size`: Input image size (default: `224`).
- `--lr-backbone`: Learning rate for the backbone.
- `--lr-fc`: Learning rate for the classification head.
- `--weight-decay`: Weight decay.

### Examples

```bash
python main.py --model resnet18 --epochs 100 --batch-size 64 --img-size 224
```

```bash
python main.py --model vittiny --epochs 100 --batch-size 128 --img-size 224 --lr-backbone 1e-3 --lr-fc 1e-2 --weight-decay 5e-4
```

> Note: `ViT-Tiny` does not support `--scratch` and must use pretrained weights.

## Model Details

### ResNet-18

- Supports ImageNet pretrained weights.
- Replaces the final fully connected layer with `Dropout + Linear` for 37 output classes.

### ViT-Tiny

- Uses `timm` to load `vit_tiny_patch16_224` pretrained weights.
- Replaces the final `head` with a linear layer for 37 classes.

## Training Workflow

Training is implemented in `Trainer`:

- Computes cross-entropy loss and backpropagation.
- Logs training loss via `swanlab`.
- Prints training loss every 10 mini-batches.
- Validates every 50 optimization steps, logging validation loss and accuracy.

## Evaluation

After training, the script evaluates the final model on the test set using `Evaluator` and prints the test accuracy.

## Saved Models

Trained weights are saved to `saved_models/` as:

- `saved_models/resnet18_pretrained.pth`
- `saved_models/resnet18_scratch.pth`
- `saved_models/vittiny_pretrained.pth`

## Notes

- The script automatically uses `cuda` if a GPU is available.
- If dataset download fails, verify network access or manually place data under `./data`.
- `swanlab` is used for experiment tracking; you can remove related calls if logging is not needed.
