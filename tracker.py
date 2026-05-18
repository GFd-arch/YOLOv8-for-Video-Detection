from pathlib import Path
import cv2
import ultralytics
from ultralytics import YOLO


class YOLOTracker:
    def __init__(
        self,
        model_root="runs/detect",
        model_ind=0,
        tracker="trackers/botsort_reid.yaml",
        save=True,
        line=None,
    ):
        self.base = Path(model_root)
        self.tracker = self._resolve_tracker_path(tracker)
        self.save = save
        self.line = line

        self.model_root = self._resolve_model_path(model_ind)

        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)


    def _resolve_model_path(self, model_ind: int) -> Path:
        if model_ind == 0:
            return self.base / "train" / "weights" / "best.pt"

        return self.base / f"train-{model_ind}" / "weights" / "best.pt"

    def _resolve_tracker_path(self, tracker: str) -> str:
        tracker_path = Path(tracker)
        if tracker_path.exists():
            return str(tracker_path)

        ultralytics_root = Path(ultralytics.__file__).resolve().parent
        builtin_dir = ultralytics_root / "cfg" / "trackers"
        candidate = builtin_dir / tracker_path.name
        if candidate.exists():
            return str(candidate)

        # fallback for botsort_reid alias
        if tracker_path.name == "botsort_reid.yaml":
            candidate = builtin_dir / "botsort.yaml"
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            f"Tracker config '{tracker}' not found locally or in Ultralyltics builtin cfg/trackers"
        )

    def _resolve_line(self, line, frame_shape):
        height, width = frame_shape[:2]
        if line is None:
            return (width // 2, 0), (width // 2, height - 1)

        if (
            isinstance(line, (tuple, list))
            and len(line) == 2
            and all(
                isinstance(point, (tuple, list)) and len(point) == 2
                for point in line
            )
        ):
            return (
                (int(line[0][0]), int(line[0][1])),
                (int(line[1][0]), int(line[1][1])),
            )

        raise ValueError(
            "line must be a tuple of two points like ((x1, y1), (x2, y2))"
        )

    def _point_side(self, point, line):
        (x1, y1), (x2, y2) = line
        x, y = point
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1e-6:
            return 0
        return 1 if cross > 0 else -1

    def _get_video_fps(self, source):
        capture = cv2.VideoCapture(str(source))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        return fps if fps and fps > 0 else 25.0

    def _annotate_frame(self, frame, line, count):
        annotated = frame.copy()
        cv2.line(annotated, line[0], line[1], (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"Line crosses: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def tracking(self, source=None, line=None):
        if source is None:
            raise ValueError("source is required for tracking")

        source_path = Path(source)
        output_name = f"{source_path.stem}_tracked"

        model = YOLO(self.model_root)

        results = model.track(
            source=str(source),
            tracker=self.tracker,
            save=False,
            project=str(self.output_dir),
            name=output_name,
            exist_ok=True,
        )

        if line is None:
            line = self.line

        line_points = None
        track_sides = {}
        crossed_ids = set()
        video_writer = None
        fps = self._get_video_fps(source)
        count = 0

        output_video = self.output_dir / f"{source_path.stem}_tracked{source_path.suffix}"
        if output_video.exists():
            output_video.unlink()

        for result in results:
            frame = result.plot()
            if line_points is None:
                line_points = self._resolve_line(line, frame.shape)

            boxes = result.boxes
            if boxes is not None and len(boxes):
                ids = boxes.id
                if ids is not None:
                    ids = ids.cpu().numpy().astype(int)
                    xyxy = boxes.xyxy
                    if xyxy is not None:
                        x_centers = ((xyxy[:, 0] + xyxy[:, 2]) / 2).cpu().numpy()
                        y_centers = ((xyxy[:, 1] + xyxy[:, 3]) / 2).cpu().numpy()
                        centers = list(zip(x_centers, y_centers))
                        for track_id, center in zip(ids, centers):
                            side = self._point_side(center, line_points)
                            last_side = track_sides.get(track_id, 0)
                            if last_side != 0 and side != 0 and side != last_side:
                                crossed_ids.add(track_id)
                            if side != 0:
                                track_sides[track_id] = side

            count = len(crossed_ids)
            annotated_frame = self._annotate_frame(frame, line_points, count)

            if self.save:
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(output_video),
                        fourcc,
                        fps,
                        (width, height),
                    )
                video_writer.write(annotated_frame)

        if video_writer is not None:
            video_writer.release()

        print(f"Tracking result saved in:")
        print(output_video.resolve())
        print(f"Total unique objects crossed the line: {count}")

        return results
    