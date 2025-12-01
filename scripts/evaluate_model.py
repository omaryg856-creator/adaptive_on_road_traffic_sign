import os
from ultralytics import YOLO

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(repo_dir)
    weights = os.environ.get("WEIGHTS_PATH", os.path.join(repo_root, "runs", "detect", "train", "weights", "best.pt"))
    if not os.path.exists(weights):
        weights = os.path.join(repo_root, "yolov8n.pt")
    data_yaml = os.environ.get("DATA_YAML", os.path.join(repo_root, "dataset", "YOLOv8", "data.yaml"))

    model = YOLO(weights)
    results = model.val(
        data=data_yaml,
        imgsz=int(os.environ.get("EVAL_IMG_SIZE", "640")),
        batch=int(os.environ.get("EVAL_BATCH", "16")),
        conf=float(os.environ.get("EVAL_CONF", "0.25")),
        iou=float(os.environ.get("EVAL_IOU", "0.7")),
        save_json=True,
        plots=True
    )
    print("Evaluation complete. Metrics saved in runs/detect/val*")

if __name__ == "__main__":
    main()
