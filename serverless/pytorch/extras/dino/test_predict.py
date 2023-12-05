from tqdm import tqdm
import supervision as sv
from pathlib import Path
import cv2
import numpy as np
from groundingdino.util.inference import Model
import argparse

def predict():
    CONFIG_PATH = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    WEIGHTS_PATH = 'weights/groundingdino_swint_ogc.pth'
    TEXT_THRESHOLD = 0.25
    BOX_THRETHOLD = 0.35
    CLASSES = ['person', 'helmet']

    model = Model(CONFIG_PATH, WEIGHTS_PATH, device='cpu')
    
    image_path = 'test.png'

    img = cv2.imread(image_path)

    detections = model.predict_with_classes(
        image=img,
        classes=CLASSES,
        box_threshold=BOX_THRETHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # drop potential detections with phrase that is not part of CLASSES set
    detections = detections[detections.class_id != None]
    # drop potential double detections
    detections = detections.with_nms()

    detections_batch = np.column_stack((
        detections.xyxy,
        detections.class_id,
        detections.confidence
    ))

    print(detections_batch)