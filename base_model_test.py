from ultralytics import YOLO
import os

repo_dir = os.path.dirname(os.path.abspath(__file__))
weights = os.environ.get(
    "WEIGHTS_PATH",
    os.path.join(repo_dir, "runs", "detect", "train", "weights", "best.pt")
)
if not os.path.exists(weights):
    weights = os.path.join(repo_dir, "yolov8n.pt")

model = YOLO(weights)
image = os.environ.get("TEST_IMAGE", os.path.join(repo_dir, "test_image.jpg"))
results = model(image)
