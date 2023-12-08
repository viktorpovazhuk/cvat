import json
import base64

import cv2
import numpy as np
from groundingdino.util.inference import Model

CONFIG_PATH = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
WEIGHTS_PATH = 'weights/groundingdino_swint_ogc.pth'

DEVICE = 'cpu'
TEXT_THRESHOLD = 0.25
BOX_THRETHOLD = 0.35
CLASSES = ['person', 'helmet']

def init_context(context):
    context.logger.info("Init context...  0%")

    model = Model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)

    context.user_data.model_handler = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run Grounding DINO model")

    data = event.body
    img_bytes = base64.b64decode(data["image"])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    threshold = float(data.get("threshold", 0.5))

    detections = context.user_data.model_handler.predict_with_classes(
        image=img,
        classes=CLASSES,
        box_threshold=BOX_THRETHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # drop potential detections with phrase that is not part of CLASSES set
    detections = detections[detections.class_id != None]
    # drop potential double detections
    detections = detections.with_nms()

    results = []
    for box, class_id, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
        label = CLASSES[int(class_id)]
        if conf >= threshold:
            results.append({
                "confidence": str(float(conf)),
                "label": label,
                "points": box.tolist(),
                "type": "rectangle",
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)