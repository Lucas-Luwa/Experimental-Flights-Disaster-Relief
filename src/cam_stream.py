import time

import cv2

from picamera2 import Picamera2, Preview
from PIL import Image
from ultralytics import YOLO


cam = Picamera2()
cam_config = cam.create_preview_configuration()
cam.configure(cam_config)
cam.start()

model = YOLO("yolov8n.yaml")

while True:
    raw_img = cam.capture_array("main")
    cv_results = model(raw_img)
    # for res in cv_results:
    #     img_arr = res.plot()
    #     img = Image.fromarray(img_arr[..., ::-1])
    img = cv_results[0].plot()
    
    # cv2.imshow("Cam", raw_img)
    cv2.imshow("Inference", img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cv2.destroyAllWindows()