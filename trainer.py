import swanlab
from ultralytics import YOLO


class YOLOTrainer:

    def __init__(self, data_yaml, params, model_name="yolov8n.pt", workers=0, val_interval=4):
        self.data_yaml = data_yaml
        self.epochs, self.imgsz, self.batch, self.device = params
        self.workers = workers
        self.val_interval = val_interval
        self.model = YOLO(model_name)

    # callback
    def on_train_epoch_end(self, trainer):

        epoch = trainer.epoch
        metrics = trainer.metrics

        # train loss
        train_box_loss = trainer.tloss[0].item()
        train_cls_loss = trainer.tloss[1].item()
        train_dfl_loss = trainer.tloss[2].item()

        swanlab.log({
            "train/box_loss": train_box_loss,
            "train/cls_loss": train_cls_loss,
            "train/dfl_loss": train_dfl_loss,
        })

        # validation metrics
        if epoch % self.val_interval == 0:

            val_box_loss = metrics.get("val/box_loss", 0)
            val_cls_loss = metrics.get("val/cls_loss", 0)
            val_dfl_loss = metrics.get("val/dfl_loss", 0)

            map50 = metrics.get("metrics/mAP50(B)", 0)
            map5095 = metrics.get("metrics/mAP50-95(B)", 0)

            swanlab.log({

                # val loss
                "val/box_loss": val_box_loss,
                "val/cls_loss": val_cls_loss,
                "val/dfl_loss": val_dfl_loss,

                # map
                "val/mAP50": map50,
                "val/mAP50-95": map5095,
            })

            print(
                f"[Val] Epoch {epoch} "
                f"mAP50={map50:.4f} "
                f"mAP50-95={map5095:.4f}"
            )

    # train
    def train(self):

        print("Start training YOLO:")

        # register callback
        self.model.add_callback(
            "on_train_epoch_end",
            self.on_train_epoch_end
        )

        results = self.model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            workers=self.workers,
        )

        print(
            "Trained model saved to: "
            "runs/detect/train*/weights/best.pt"
        )

        return results


# evaluator
class YOLOEvaluator:
    def __init__(self, model_path, data_yaml, device=0):

        self.data_yaml = data_yaml
        self.device = device
        self.model = YOLO(model_path)

    def evaluate(self):
        print("Start evaluation:")
        metrics = self.model.val(
            data=self.data_yaml,
            device=self.device,
        )
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")

        return metrics
    