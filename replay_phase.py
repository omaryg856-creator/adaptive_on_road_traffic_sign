from ultralytics import YOLO
import geocoder
import json
import math
import os

repo_dir = os.path.dirname(os.path.abspath(__file__))
MEM_FILE = os.environ.get("SIGN_MEMORY", os.path.join(repo_dir, "sign_memory.json"))

if os.path.exists(MEM_FILE):
    with open(MEM_FILE, "r") as f:
        memory = json.load(f)
else:
    memory = []

weights = os.environ.get(
    "WEIGHTS_PATH",
    os.path.join(repo_dir, "runs", "detect", "train", "weights", "best.pt")
)
if not os.path.exists(weights):
    weights = os.path.join(repo_dir, "yolov8n.pt")
model = YOLO(weights)

def get_gps():
    loc = geocoder.ip("me")
    return loc.latlng

def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3
    phi1 = lat1 * math.pi/180
    phi2 = lat2 * math.pi/180
    dphi = (lat2-lat1) * math.pi/180
    dlambda = (lon2-lon1) * math.pi/180
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2*math.atan2(math.sqrt(a), math.sqrt(1-a)))

def find_predicted_signs(lat, lon, radius=40):
    predicted = []
    for entry in memory:
        d = haversine(lat, lon, entry["latitude"], entry["longitude"])
        if d <= radius:
            predicted.append(entry)
    return predicted

def detect_current_sign(image_path):
    results = model(image_path)[0]
    detected = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        conf = float(box.conf[0])
        detected.append(label)
    return detected

def replay_phase(image_path):
    lat, lon = get_gps()
    predicted = find_predicted_signs(lat, lon)
    detected_now = detect_current_sign(image_path)

    print("Predicted signs near you:")
    for s in predicted:
        print(s["sign_type"])

    print("\nYOLO detected:")
    print(detected_now)

    results = []
    for p in predicted:
        expected = p["sign_type"]
        if expected in detected_now:
            results.append((expected, "Match"))
        else:
            results.append((expected, "Missing"))

    for d in detected_now:
        if all(d != p["sign_type"] for p in predicted):
            results.append((d, "New Sign"))

    return results

if __name__ == "__main__":
    image_path = os.environ.get("REPLAY_IMAGE", os.path.join(repo_dir, "test_image.jpg"))
    results = replay_phase(image_path)

    print("\nReplay Phase Results:")
    for r in results:
        print(r)
