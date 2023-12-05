from ultralytics import YOLO
from pathlib import Path

CLASSES = ['person']
MODEL_PATH = 'person_close.pt'
DEVICE = 'cpu'
BOX_THRESHOLD = 0.35

model = YOLO(MODEL_PATH)

res = model.predict('test.png', imgsz=640, device=DEVICE, conf=BOX_THRESHOLD)[0]

detections = res.cpu().numpy().boxes

for box, class_id, conf in zip(detections.xyxy, detections.cls, detections.conf):
    label = CLASSES[class_id]
    print({"confidence": str(conf),
            "label": label,
            "points": box.tolist(),
            "type": "rectangle"})