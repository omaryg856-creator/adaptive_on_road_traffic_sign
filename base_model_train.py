from ultralytics import YOLO
import os

def main():
    model = YOLO("yolov8n.pt")
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.environ.get("DATA_YAML", os.path.join(repo_dir, "dataset", "YOLOv8", "data.yaml"))
    
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        workers=0
    )
    print("Done")

if __name__ == "__main__":
    main()
