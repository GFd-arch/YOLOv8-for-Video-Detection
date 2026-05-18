# VisDrone YOLO Tracking Project

This repository contains a YOLO-based object detection and tracking pipeline built for the VisDrone dataset. It includes dataset preparation, YOLO training, video tracking with object IDs, and a line-crossing counting feature for tracking results.

## Features

- Download and prepare the VisDrone dataset
- Convert VisDrone annotations to YOLO format
- Train YOLO models (`yolov8n`, `yolo26n`) on VisDrone data
- Track objects in video with Ultralyitcs tracker
- Count unique objects crossing a virtual line during video tracking
- Save annotated tracking videos to `outputs/`

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The project also depends on:

- `torch` (choose the appropriate CPU or CUDA build for your environment)
- `swanlab`
- `kagglehub`

## Project Structure

- `dataset.py` - prepare and convert VisDrone dataset to YOLO format
- `trainer.py` - YOLO training and evaluation helpers
- `tracker.py` - video tracking and line-crossing count logic
- `main.py` - command-line entry point for training and tracking
- `data/` - raw dataset and converted YOLO dataset
- `videos/` - source videos for tracking
- `outputs/` - generated tracking videos and results

## Dataset Preparation

1. Place the VisDrone dataset under `data/raw/`.
2. Run the dataset conversion pipeline from `main.py`, which also prepares `visdrone.yaml`.

The pipeline automatically downloads the dataset via Kaggle if it is not already present.

## Training

Use `main.py` to train a YOLO model on the converted VisDrone dataset.

```bash
python main.py --train --model yolov8n --epoch 100 --batchsize 8 --img-size 640
```

This will:

- prepare the dataset
- generate `visdrone.yaml`
- train the model using Ultralyitcs YOLO
- save best weights under `runs/detect/train*/weights/best.pt`

## Tracking

Use `main.py` to run object tracking on a video file.

```bash
python main.py --track --model yolov8n --model-id 5 --video-source videos/video1.mp4
```

This will:

- load the trained model weights from `runs/detect/train-5/weights/best.pt`
- run tracking on the provided video
- save the annotated video to `outputs/<video_name>_tracked.mp4`

## Line Crossing Count

The tracking module includes a line-crossing counter. By default, it draws a horizontal center line on the video and counts unique tracked objects that cross the line.

For custom usage, call `YOLOTracker` directly:

```python
from tracker import YOLOTracker

yolotrack = YOLOTracker(model_ind=5, save=True)
yolotrack.tracking(
    source="videos/real_strengthed.mp4",
    line=((100, 200), (540, 200))
)
```

## Output

Tracked videos are saved in the `outputs/` directory. Each frame is annotated with:

- tracked object boxes and IDs
- the virtual line
- the current count of unique objects that crossed the line

## Notes

- The project is designed for the VisDrone dataset, but the YOLO conversion and tracking pipeline can be adapted for other datasets with similar format.
- If you want to use a different trained model ID, update `--model-id` to match the saved training run.
