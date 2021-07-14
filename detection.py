from imageai.Detection import ObjectDetection
import os

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image="imgs/rgb.png", output_image_path="igms/out.png")
print(len(detections))
for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("x1 y1 - lower left; x2 y2 - upper right")