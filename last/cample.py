import numpy as np
import pybullet as pb
import pybullet_data
import time

physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = pb.loadURDF('plane.urdf')

visualShapeId = pb.createVisualShape(
    shapeType=pb.GEOM_MESH,
    fileName='random_urdfs/000/000.obj',
    rgbaColor=None,
    meshScale=[0.1, 0.1, 0.1])

collisionShapeId = pb.createCollisionShape(
    shapeType=pb.GEOM_MESH,
    fileName='random_urdfs/000/000_coll.obj',
    meshScale=[0.1, 0.1, 0.1])

multiBodyId = pb.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=collisionShapeId, 
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0, 0, 1],
    baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

for i in range (10000):
    pb.stepSimulation()
    time.sleep(1./240.)


import os, glob, random
texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)
random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
textureId = pb.loadTexture(random_texture_path)
pb.changeVisualShape(multiBodyId, -1, textureUniqueId=textureId)

pb.setGravity(0, 0, -9.8)
pb.setRealTimeSimulation(1)

viewMatrix = pb.computeViewMatrix(
    cameraEyePosition=[0, 0, 3],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 1, 0])

projectionMatrix = pb.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=3.1)

width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
    width=224, 
    height=224,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix)
