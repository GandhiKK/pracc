import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from imageai.Detection import ObjectDetection

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("models/yolo.h5")
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image="imgs/rgb.png", output_image_path="imgs/out.png")
print(len(detections))
for eachObject in detections:
    if eachObject["name"] == 'sports ball':
        print(eachObject["box_points"])
        print("x1 y1 - lower left; x2 y2 - upper right")
