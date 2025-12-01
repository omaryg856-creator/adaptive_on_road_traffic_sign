from ultralytics import YOLO
import geocoder
import folium
import json
import os
import datetime
import math

# Load trained YOLO model
repo_dir = os.path.dirname(os.path.abspath(__file__))
weights = os.environ.get(
    "WEIGHTS_PATH",
    os.path.join(repo_dir, "runs", "detect", "train", "weights", "best.pt")
)
if not os.path.exists(weights):
    weights = os.path.join(repo_dir, "yolov8n.pt")
model = YOLO(weights)

# Memory file
MEM_FILE = os.environ.get("SIGN_MEMORY", os.path.join(repo_dir, "sign_memory.json"))

# Load existing memory or create new
if os.path.exists(MEM_FILE):
    with open(MEM_FILE, "r") as f:
        memory_data = json.load(f)
else:
    memory_data = []

def get_gps():
    loc = geocoder.ip("me")
    return loc.latlng  # returns (lat, lon)

def save_memory(entry):
    updated = False
    try:
        lat = entry.get("latitude")
        lon = entry.get("longitude")
        label = entry.get("sign_type")
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371e3
            phi1 = lat1 * math.pi/180
            phi2 = lat2 * math.pi/180
            dphi = (lat2-lat1) * math.pi/180
            dlambda = (lon2-lon1) * math.pi/180
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            return R * (2*math.atan2(math.sqrt(a), math.sqrt(1-a)))
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
    with open(MEM_FILE, "w") as f:
        json.dump(memory_data, f, indent=4)

def update_map(lat, lon, label):
    # Center map on Michigan
    michigan_center = [44.3148, -85.6024]  # Michigan center point

    mymap = folium.Map(location=[lat, lon], zoom_start=14)

    # Add detected sign marker
    folium.Marker(
        [lat, lon],
        popup=f"Detected: {label}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(mymap)

    # Save map
    mymap.save("sign_map.html")
    print("Map updated → sign_map.html")

def detect_sign_and_map(image_path):
    print("🔍 Running YOLO detection...")
    results = model(image_path)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        mapping_path = os.path.join(repo_dir, "class_names.json")
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r") as mf:
                    class_map = json.load(mf)
                label = class_map.get(label, label)
            except Exception:
                pass
        conf = float(box.conf[0])

        print(f"✔ Detected sign: {label} ({conf:.2f})")

        # Get GPS
        lat, lon = get_gps()
        print(f"📍 GPS Location: {lat}, {lon}")

        # Save in memory
        entry = {
            "sign_type": label,
            "confidence": conf,
            "latitude": lat,
            "longitude": lon,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        save_memory(entry)

        # Update map
        update_map(lat, lon, label)

if __name__ == "__main__":
    image_path = os.environ.get("IMAGE_PATH", os.path.join(repo_dir, "test_image.jpg"))
    detect_sign_and_map(image_path)
