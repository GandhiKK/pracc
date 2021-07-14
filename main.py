import numpy as np
import pybullet as pb
import time
import pybullet_data
import random
import matplotlib.pyplot as plt

DURATION = 10000
FORCE = 50

physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -10)
planeId = pb.loadURDF("plane100.urdf")

cubeStartPos = [0, 0, 0.3]
cubeStartOrientation = pb.getQuaternionFromEuler([0, 0, 0])
boxId = pb.loadURDF("soccerball.urdf", cubeStartPos, cubeStartOrientation, globalScaling=0.5)


for i in range(DURATION):
    if (i % 100 == 0):
        xpos, ypos = random.randint(-5,5), random.randint(-5,5)
    pb.stepSimulation()
    time.sleep(1./240.)
    boxPos, boxOrn = pb.getBasePositionAndOrientation(boxId)
    force = FORCE * (np.array([xpos, ypos, 1]) - np.array(boxPos))
    # pb.applyExternalForce(objectUniqueId=boxId, linkIndex=-1, forceObj=force, posObj=boxPos, flags=pb.WORLD_FRAME)

    viewMatrix = pb.computeViewMatrix(
        # cameraEyePosition=[boxPos[0], boxPos[1], 10], # физическое расположение камеры
        cameraEyePosition=[0, 0, 5], # физическое расположение камеры
        cameraTargetPosition=[0, 0, 0], # куда смотрит
        cameraUpVector=[0, 1, 0])

    projectionMatrix = pb.computeProjectionMatrixFOV(
        fov=45.0, # типа ширь камеры
        aspect=1.0,
        nearVal=0.1,
        farVal=5.1)
    
    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
        width=128, 
        height=128,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)
    
    # if (i % 100 == 0):
    if (i == 100):
        rgb_opengl = np.reshape(rgbImg, (height, width, 4)) * 1. / 255.
        seg_opengl = np.reshape(segImg, [width, height]) * 1. / 255.
        
        # print(rgb_opengl.shape)
        # print(seg_opengl.shape)

        # plt.subplot(1, 2, 1)    
        plt.imshow(rgb_opengl)
        plt.title('RGB')    
        
        plt.savefig(f'imgs/rgb.png')

        # plt.subplot(1, 2, 2)    
        plt.imshow(seg_opengl)
        plt.title('Seg')   
        
        plt.savefig(f'imgs/seq.png')

        # plt.show()
    
pb.disconnect()