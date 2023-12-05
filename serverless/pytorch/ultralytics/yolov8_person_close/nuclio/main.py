import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ['person']
MODEL_PATH = 'person_close.pt'

DEVICE = 'cpu'
BOX_THRESHOLD = 0.35

def init_context(context):
    context.logger.info("Init context...  0%")

    model = YOLO(MODEL_PATH)

    context.user_data.model_handler = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run Grounding DINO model")

    data = event.body
    img_bytes = base64.b64decode(data["image"])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    threshold = float(data.get("threshold", 0.5))

    res = context.user_data.model_handler.predict(img, imgsz=640, device=DEVICE, conf=BOX_THRESHOLD)[0]
    detections = res.cpu().numpy().boxes

    results = []
    for box, class_id, conf in zip(detections.xyxy, detections.cls, detections.conf):
        label = CLASSES[class_id]
        if conf >= threshold:
            results.append({
                "confidence": str(conf),
                "label": label,
                "points": box.tolist(),
                "type": "rectangle",
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)