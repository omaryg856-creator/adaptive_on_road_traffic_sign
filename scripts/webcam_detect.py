import os
import json
import math
import datetime
import cv2
import geocoder
from ultralytics import YOLO

repo_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(repo_dir)
weights = os.environ.get("WEIGHTS_PATH", os.path.join(repo_root, "runs", "detect", "train", "weights", "best.pt"))
if not os.path.exists(weights):
    weights = os.path.join(repo_root, "yolov8n.pt")
model = YOLO(weights)
mem_file = os.environ.get("SIGN_MEMORY", os.path.join(repo_root, "sign_memory.json"))
if os.path.exists(mem_file):
    try:
        with open(mem_file, "r") as f:
            memory_data = json.load(f)
    except Exception:
        memory_data = []
else:
    memory_data = []

mapping_path = os.path.join(repo_root, "class_names.json")
class_map = {}
if os.path.exists(mapping_path):
    try:
        with open(mapping_path, "r") as mf:
            class_map = json.load(mf)
    except Exception:
        class_map = {}

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

def find_predicted_signs(lat, lon, radius=40.0):
    predicted = []
    for m in memory_data:
        try:
            d = haversine(lat, lon, m.get("latitude"), m.get("longitude"))
            if d <= radius:
                predicted.append(m)
        except Exception:
            continue
    return predicted

def save_memory(entry):
    updated = False
    try:
        lat = entry.get("latitude")
        lon = entry.get("longitude")
        label = entry.get("sign_type")
        for m in memory_data:
            if m.get("sign_type") == label:
                d = haversine(lat, lon, m.get("latitude"), m.get("longitude"))
                if d <= float(os.environ.get("MEM_UPDATE_RADIUS", "30")):
                    alpha = float(os.environ.get("MEM_UPDATE_ALPHA", "0.3"))
                    m["latitude"] = alpha*lat + (1-alpha)*m.get("latitude")
                    m["longitude"] = alpha*lon + (1-alpha)*m.get("longitude")
                    m["confidence"] = max(m.get("confidence", 0), entry.get("confidence", 0))
                    m["timestamp"] = entry.get("timestamp")
                    updated = True
                    break
    except Exception:
        pass
    if not updated:
        memory_data.append(entry)
    with open(mem_file, "w") as f:
        json.dump(memory_data, f, indent=4)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Unable to open webcam")
    frame_idx = 0
    skip_pred = int(os.environ.get("DETECT_SKIP_PRED", "5"))
    skip_new = int(os.environ.get("DETECT_SKIP_NEW", "1"))
    last_gps = None
    last_gps_time = 0.0
    gps_interval = float(os.environ.get("GPS_INTERVAL_SEC", "5"))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = datetime.datetime.utcnow().timestamp()
        if last_gps is None or now - last_gps_time >= gps_interval:
            try:
                last_gps = get_gps()
                last_gps_time = now
            except Exception:
                last_gps = (0.0, 0.0)
        lat, lon = last_gps or (0.0, 0.0)
        predicted = find_predicted_signs(lat, lon)
        do_detect = False
        if predicted:
            do_detect = (frame_idx % skip_pred == 0)
        else:
            do_detect = (frame_idx % skip_new == 0)
        detections = []
        if do_detect:
            r = model.predict(source=frame, verbose=False)[0]
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                label = class_map.get(label, label)
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append((label, conf, xyxy))
                lt, rb = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(frame, lt, rb, (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (lt[0], max(0, lt[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                entry = {
                    "sign_type": label,
                    "confidence": conf,
                    "latitude": lat,
                    "longitude": lon,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                }
                save_memory(entry)
        cv2.imshow("webcam_detect", frame)
        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
