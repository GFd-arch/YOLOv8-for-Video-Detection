import shutil, os, zipfile, cv2
import yaml
from tqdm import tqdm
from pathlib import Path
import kagglehub
import shutil


class VisDroneDataset:
    def __init__(self, save_dir="data"):
        save_dir = Path(save_dir)
        self.save_dir = save_dir / "raw"
        self.yolo_dir = save_dir / "visdrone_yolo"    

    
    def get_data(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = self.save_dir / "kaggle_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir.resolve())

        # 检查是否已经存在数据
        train_dir = self.save_dir / "VisDrone2019-DET-train"
        val_dir = self.save_dir / "VisDrone2019-DET-val"

        if train_dir.exists() and val_dir.exists():
            print("VisDrone dataset already exists.")
            print(f"Dataset ready at: {self.save_dir}")
            return

        print("Downloading VisDrone dataset...")

        # 下载数据集
        dataset_path = kagglehub.dataset_download(
            "banuprasadb/visdrone-dataset"
        )

        dataset_path = Path(dataset_path)

        print(f"Downloaded to cache: {dataset_path}")

        # 拷贝到 data/raw
        # 拷贝到 data/raw
        for item in dataset_path.iterdir():

            # 如果是 VisDrone_Dataset，则展开里面内容
            if item.is_dir() and item.name == "VisDrone_Dataset":

                for sub_item in item.iterdir():
                    target = self.save_dir / sub_item.name

                    if target.exists():
                        continue

                    if sub_item.is_dir():
                        shutil.copytree(sub_item, target)
                    else:
                        shutil.copy2(sub_item, target)

            else:
                target = self.save_dir / item.name

                if target.exists():
                    continue

                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)

        print(f"Dataset ready at: {self.save_dir}")
        


    def _find_split_root(self, split):
        split = split.lower()

        candidates = [
            self.save_dir / f"VisDrone2019-DET-{split}",
            self.save_dir / split,
            self.save_dir / split.capitalize(),
            self.save_dir / split.upper(),
        ]

        # 自动搜索所有可能目录
        candidates.extend(
            [p for p in self.save_dir.rglob("*") if p.is_dir()]
        )

        for root in candidates:
            img_dir = root / "images"
            ann_dir = root / "labels"

            if img_dir.exists() and ann_dir.exists():
                # 判断当前 root 是否对应 split
                root_name = root.name.lower()

                if split in root_name:
                    return img_dir, ann_dir

        raise FileNotFoundError(
            f"{split} images/labels not found in {self.save_dir}"
        )
    

    def _visdrone_line_to_yolo(self, line, img_w, img_h):
        line = line.strip()
        if not line:
            return None

        if "," in line:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                return None
            try:
                x, y, w, h = map(float, parts[0:4])
                category = int(float(parts[5]))
            except ValueError:
                return None

            # 忽略无效框
            if category <= 0 or w <= 1 or h <= 1:
                return None

            xc = (x + w / 2.0) / img_w
            yc = (y + h / 2.0) / img_h
            ww = w / img_w
            hh = h / img_h
            cls = category - 1
            return cls, xc, yc, ww, hh

        parts = line.split()
        if len(parts) < 5:
            return None
        try:
            cls = int(float(parts[0]))
            xc, yc, ww, hh = map(float, parts[1:5])
        except ValueError:
            return None

        if ww <= 0 or hh <= 0:
            return None

        return cls, xc, yc, ww, hh
    

    def _get_image_size(self, img_path):
        img = cv2.imread(str(img_path))

        if img is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {img_path}")
        h, w = img.shape[:2]
        return w, h
        

    def convert_split(self, split):
        img_dir, ann_dir = self._find_split_root(split)

        out_img_dir = self.yolo_dir / "images" / split
        out_lbl_dir = self.yolo_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        ann_files = sorted(ann_dir.glob("*.txt"))
        if not ann_files:
            raise FileNotFoundError(f"No annotation files found in {ann_dir}.")
        
        for ann_path in tqdm(ann_files, desc=f"Convert {split}"):
            stem = ann_path.stem
            img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                png = img_dir / f"{stem}.png"
                if png.exists():
                    img_path = png
                else:
                    continue

            img_w, img_h = self._get_image_size(img_path)

            yolo_lines = []
            with ann_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    converted = self._visdrone_line_to_yolo(line, img_w, img_h)
                    if converted is None:
                        continue
                    cls, xc, yc, ww, hh = converted
                    yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

            # labels: write empty file even without bbox, for YOLO compatibility
            (out_lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            # images: copy
            out_img_path = out_img_dir / img_path.name
            if not out_img_path.exists():
                out_img_path.write_bytes(img_path.read_bytes())


    def write_yaml(self, yaml_path):
        names = [
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
        ]
        cfg = {
            "path": str(self.yolo_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: n for i, n in enumerate(names)},
        }
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


    def to_YOLO(self):
        self.yolo_dir.mkdir(exist_ok=True)

        self.convert_split("train")
        self.convert_split("val")

        yaml_path = Path("visdrone.yaml").resolve()
        self.write_yaml(yaml_path)

        print(f"Converted successfully:{self.yolo_dir.resolve()}")
        print(f"Generated YAML: {yaml_path}")
        