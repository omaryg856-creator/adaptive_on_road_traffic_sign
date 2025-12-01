import os
import xml.etree.ElementTree as ET

# Paths
repo_dir = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("DATASET_BASE", os.path.join(repo_dir, "dataset", "challenging-dev", "challenging"))
XML_FOLDER = os.path.join(BASE, "Annotations")
IMG_FOLDER = os.path.join(BASE, "JPEGImages")
SET_FOLDER = os.path.join(BASE, "ImageSets", "Main")

OUTPUT = os.path.join(BASE, "YOLOv8")
os.makedirs(OUTPUT, exist_ok=True)

def convert_bbox(size, box):
    w, h = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    return x_center, y_center, width, height

def convert_xml_to_yolo(xml_path, label_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()

        # Convert class name (string) into YOLO class index
        if name not in label_map:
            label_map[name] = len(label_map)
        cls_id = label_map[name]

        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))

        xc, yc, w, h = convert_bbox((img_w, img_h), (xmin, ymin, xmax, ymax))
        lines.append(f"{cls_id} {xc} {yc} {w} {h}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

def process_split(split):
    out_img = os.path.join(OUTPUT, split, "images")
    out_lbl = os.path.join(OUTPUT, split, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    list_file = os.path.join(SET_FOLDER, f"{split}.txt")

    with open(list_file, "r") as f:
        names = [x.strip() for x in f.readlines()]

    for n in names:
        xml = os.path.join(XML_FOLDER, f"{n}.xml")
        img = os.path.join(IMG_FOLDER, f"{n}.jpg")

        if not os.path.exists(xml) or not os.path.exists(img):
            continue

        out_img_path = os.path.join(out_img, f"{n}.jpg")
        out_lbl_path = os.path.join(out_lbl, f"{n}.txt")

        with open(img, "rb") as f1, open(out_img_path, "wb") as f2:
            f2.write(f1.read())

        convert_xml_to_yolo(xml, out_lbl_path)

label_map = {}

for split in ["train", "val", "test"]:
    process_split(split)

# Create data.yaml
yaml_path = os.path.join(OUTPUT, "data.yaml")
with open(yaml_path, "w") as f:
    f.write("train: train/images\n")
    f.write("val: val/images\n")
    f.write("test: test/images\n")
    f.write(f"nc: {len(label_map)}\n")
    f.write("names: [\n")
    for k in label_map:
        f.write(f"  '{k}',\n")
    f.write("]\n")

print("Conversion finished successfully!")
print(f"Classes found: {len(label_map)}")
