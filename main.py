import argparse
from pathlib import Path

import torch
import swanlab

from dataset import VisDroneDataset
from trainer import YOLOTrainer, YOLOEvaluator
from tracker import YOLOTracker



# train
def train_YOLO(model_name="yolov8n", epochs=100, batchsize=8, imgsz=640):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # dataset
    ds = VisDroneDataset()

    ds.get_data()
    ds.to_YOLO()

    data_yaml = str(
        Path("visdrone.yaml").resolve()
    )

    train_paras = [
        epochs,
        imgsz,
        batchsize,
        device
    ]

    yolotrain = YOLOTrainer(
        data_yaml,
        params=train_paras,
        model_name=model_name
    )

    yolotrain.train()


# tracking
def track_video(model_name="yolov8n", model_id=5, video_source="videos/real_strengthed.mp4"):

    video_path = Path(video_source)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_source}"
        )

    print(f"Tracking video: {video_source}")

    yolotrack = YOLOTracker(
        model_ind=model_id
    )

    yolotrack.tracking(
        source=str(video_path)
    )


# main
def main():

    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train YOLO model"
    )

    parser.add_argument(
        "--track",
        action="store_true",
        help="Track objects in video"
    )

    # model
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolo26n"],
        help="YOLO model type"
    )

    # train params
    parser.add_argument(
        "--epoch",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        default=8
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640
    )

    # tracking params
    parser.add_argument(
        "--model-id",
        type=int,
        default=5,
        help="Saved model id"
    )

    parser.add_argument(
        "--video-source",
        type=str,
        default="videos/real_strengthed.mp4"
    )

    args = parser.parse_args()


    
    # train
    if args.train:

        swanlab.init(
            project="visdrone-yolo",
            experiment_name=f"{args.model}_e{args.epoch}_b{args.batchsize}"
        )

        train_YOLO(
            model_name=args.model,
            epochs=args.epoch,
            batchsize=args.batchsize,
            imgsz=args.img_size
        )

        swanlab.finish()

    # track
    if args.track:

        track_video(
            model_name=args.model,
            model_id=args.model_id,
            video_source=args.video_source
        )

    # no mode selected
    if not args.train and not args.track:

        print("Please specify --train or --track")


# run
if __name__ == "__main__":
    main()