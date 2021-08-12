import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

im = cv2.imread('imgs/rgb.png')

bbox, label, conf = cv.detect_common_objects(im)

print(bbox, label, conf)

output_image = draw_bbox(im, bbox, label, conf)

print(bbox)
print(bbox[0][0])

# plt.imshow(output_image)
# plt.show()