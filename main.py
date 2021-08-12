import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pybullet as pb
import time
import pybullet_data
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.preprocessing import Normalizer
import cvlib as cv
from cvlib.object_detection import draw_bbox
from imageai.Detection import ObjectDetection

DURATION = 100000
FORCE = 30

ideal_coords = []
horz_shift = 0
vert_shift = 0
z_shift = 5

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("models/yolo.h5")
detector.loadModel()

physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -10)
planeId = pb.loadURDF("plane100.urdf")

cubeStartPos = [0, 0, 0.3]
cubeStartOrientation = pb.getQuaternionFromEuler([0, 0, 0])
boxId = pb.loadURDF("soccerball.urdf", cubeStartPos, cubeStartOrientation, globalScaling=0.5)





# def detect():
#     # start_time = time.time()
#     detections = detector.detectObjectsFromImage(input_image="imgs/rgb.png", output_image_path="imgs/out.png")
#     boxl = [0, 0, 0, 0]
#     for eachObject in detections:
#         if eachObject["name"] == 'sports ball':
#             boxl = eachObject["box_points"]
#             box = f'[{boxl[0]} {boxl[1]} {boxl[2]} {boxl[3]}]'
#             print(box)
#             pb.addUserDebugText(box, [0, 0, 2], [0, 0, 0], lifeTime=3)
#             # GLDebugDrawString(xStart,yStart,text);
#             # print("x1 y1 - lower left; x2 y2 - upper right")
#     # print("--- %s seconds ---" % (time.time() - start_time))
#     return len(detections), boxl


# def detect2():
#     # start_time = time.time()
#     ind = 0
#     im = cv2.imread('imgs/rgb.png')
#     bbox, label, conf = cv.detect_common_objects(im)
    
#     if len(bbox) == 0:
#         return len(bbox), bbox
    
#     if 'sports ball' not in str(label):
#         return 0, bbox
    
#     for i in range(len(label)):
#         if label[i] == 'sports ball':
#             ind = i
#             break
#     # print(bbox)
#     # print(label)
#     print(bbox[ind])
    
#     output_image = draw_bbox(im, bbox, label, conf)
#     plt.imshow(output_image)
#     plt.title('out')  
#     plt.savefig(f'imgs/out1.png')
#     plt.clf()
#     box = f'[{bbox[ind][0]} {bbox[ind][1]} {bbox[ind][2]} {bbox[ind][3]}]'
#     pb.addUserDebugText(box, [0, 0, 2], [0, 0, 0], lifeTime=3)
#     # print("--- %s seconds ---" % (time.time() - start_time))
#     return len(bbox), bbox[ind]
    


# def rotate(box):
#     center_ideal_horz = (ideal_coords[2] - ideal_coords[0]) + ideal_coords[0]
#     center_ideal_vert = (ideal_coords[3] - ideal_coords[1]) + ideal_coords[1]    
#     center_horz = (box[2] - box[0]) + box[0]
#     center_vert = (box[3] - box[1]) + box[1]
    
#     diff_horz = center_ideal_horz - center_horz
#     diff_vert = center_ideal_vert - center_vert
    
#     return diff_horz, diff_vert


# def dist_to_obj(P_hat):
#     # W - ширина (м)
#     # P - ширина (pix)
#     # D - расстояние до объекта
#     # F = (P * D) / W
#     W = 0.5
#     F = 630
#     D_hat = (W * F) / P_hat
#     print(D_hat)
#     return 5 - D_hat


# def set_camera(pos=np.ones(3), target=np.zeros(3)):
#     disp = target - pos
#     disp = disp.reshape(-1, 1)
#     transformer = Normalizer().fit(disp)
#     dist = transformer.transform(disp)
#     yaw = np.arctan2(-disp[0],disp[1]) * 180/np.pi
#     pitch = np.arctan2(disp[2],np.sqrt(disp[0]**2+disp[1]**2)) * 180/np.pi
#     pb.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target.tolist())                       

   
# boxId2 = pb.loadURDF("objs/quad.urdf", [4, 4, 5.5], pb.getQuaternionFromEuler([0, 0, 0]), globalScaling=2.5)   
   
# for i in range(DURATION):
#     if (i % 100 == 0):
#         xpos, ypos = random.randint(-5,5), random.randint(-5,5)
#     pb.stepSimulation()
#     time.sleep(1./240.)
#     boxPos, boxOrn = pb.getBasePositionAndOrientation(boxId)
#     force = FORCE * (np.array([xpos, ypos, 1]) - np.array(boxPos))
#     pb.applyExternalForce(objectUniqueId=boxId, linkIndex=-1, forceObj=force, posObj=boxPos, flags=pb.WORLD_FRAME)
#     boxPos2, boxOrn2 = pb.getBasePositionAndOrientation(boxId)


#     viewMatrix = pb.computeViewMatrix(
#         # cameraEyePosition=[boxPos[0], boxPos[1], 10], # физическое расположение камеры
#         cameraEyePosition=[0+(horz_shift / 128), 0+(vert_shift / 128), z_shift], # физическое расположение камеры
#         cameraTargetPosition=[0+(horz_shift / 128), 0+(vert_shift / 128), 0], # куда смотрит
#         cameraUpVector=[0, 1, 0])

#     # pb.resetBasePositionAndOrientation(boxId2, [boxPos2[0], boxPos2[1], 5.5], pb.getQuaternionFromEuler([0, 0, 0]))
#     pb.resetBasePositionAndOrientation(boxId2, [horz_shift / 128, vert_shift / 128, z_shift+0.5], pb.getQuaternionFromEuler([0, 0, 0]))
    
#     projectionMatrix = pb.computeProjectionMatrixFOV(
#         fov=35.0, # типа ширь камеры
#         aspect=1.0,
#         nearVal=0.1,
#         farVal=z_shift+0.1)
    
#     width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
#         width=128, 
#         height=128,
#         viewMatrix=viewMatrix,
#         projectionMatrix=projectionMatrix)
    
    
#     if (i % 3 == 0):
        
        
        
#         rgb_opengl = np.reshape(rgbImg, (height, width, 4)) * 1. / 255.
#         # seg_opengl = np.reshape(segImg, [width, height]) * 1. / 255.
          
#         plt.imshow(rgb_opengl)
#         plt.title('RGB')           
#         plt.savefig(f'imgs/rgb.png')
#         # plt.show()
        
#         num_obj, coords = detect2()
        
#         if i == 0:
#             ideal_coords = coords
        
#         if num_obj == 0:
#             # i -= 1
#             continue     
                     
#         horz_shift1, vert_shift1 = rotate(coords)
#         horz_shift -= horz_shift1
#         vert_shift += vert_shift1
                
#         print(horz_shift/128, vert_shift/128)
        
#         # set distance
#         # if (i % 5 == 0):
#         z_shift1 = dist_to_obj(coords[2] - coords[0])
#         z_shift += z_shift1
        
#             # print(z_shift1)
        
        
#         # set_camera(np.array([(horz_shift / 128) - 2, (vert_shift / 128), 5]), 
#         #            np.array([horz_shift / 128, vert_shift / 128, 0]))
        
        
#         plt.clf()
#         img_out = Image.open('imgs/out.png')
#         plt.imshow(img_out)
#         # plt.show()
        
    
# pb.disconnect()