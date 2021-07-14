import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF('plane.urdf')
cubeStartPos = [2,0,3]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

print(cubeStartOrientation)

# boxId = p.loadURDF('r2d2.urdf',cubeStartPos, cubeStartOrientation)

shift = [0, -0.02, 0]
meshScale = [0.1, 0.1, 0.1]

visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName="duck.obj",
                                    rgbaColor=[1, 1, 1, 1],
                                    specularColor=[0.4, .4, 0],
                                    visualFramePosition=shift,
                                    meshScale=meshScale)
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName="duck_vhacd.obj",
                                          collisionFramePosition=shift,
                                          meshScale=meshScale)


# rangex = 3
# rangey = 3
# for i in range(rangex):
#   for j in range(rangey):
p.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=[0, 0, 0],
                      useMaximalCoordinates=True)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)

p.disconnect()