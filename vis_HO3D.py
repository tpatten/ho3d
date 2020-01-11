"""
Visualize the projections in published HO-3D dataset
"""
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

baseDir = './'


seqName = 'MC1'

fileID = ['0025']


def lineParser(line, annoDict):
    '''
    Parses a line in the 'anno.txt' and creates a entry in dict with id as key
    :param line: line from 'anno.txt'
    :param annoDict: dict in which an entry should be added
    :return:
    '''
    lineList = line.split()
    id = lineList[0]
    objID = lineList[1]
    paramsList = list(map(float, lineList[2:]))

    assert id not in annoDict.keys(), 'Something wrong with the annotation file...'

    annoDict[id]= {
        'objID': objID,
        'handJoints': np.reshape(np.array(paramsList[:63]), [21,3]),
        'handPose': np.array(paramsList[63:63 + 48]),
        'handTrans': np.array(paramsList[63 + 48:63 + 48 + 3]),
        'handBeta': np.array(paramsList[63 + 48 + 3: 63 + 48 + 3 + 10]),
        'objRot': np.array(paramsList[63 + 48 + 3 + 10: 63 + 48 + 3 + 10 + 3]),
        'objTrans': np.array(paramsList[63 + 48 + 3 + 10 + 3: 63 + 48 + 3 + 10 + 3 + 3])
        }


    return annoDict

def parseAnnoTxt(filename):
    '''
    Parse the 'anno.txt'
    :param filename: path to 'anno.txt'
    :return: dict with id as keys
    '''
    ftxt = open(filename, 'r')
    annoLines = ftxt.readlines()
    annoDict = {}
    for line in annoLines:
        lineParser(line, annoDict)

    return annoDict

def decodeDepthImg(inFileName, dsize=None):
    '''
    Decode the depth image to depth map in meters
    :param inFileName: input file name
    :return: depth map (float) in meters
    '''
    depthScale = 0.00012498664727900177
    depthImg = cv2.imread(inFileName)
    if dsize is not None:
        depthImg = cv2.resize(depthImg, dsize, interpolation=cv2.INTER_CUBIC)

    dpt = depthImg[:, :, 0] + depthImg[:, :, 1] * 256
    dpt = dpt * depthScale

    return dpt

def project3DPoints(camMat, pts3D, isOpenGLCoords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        pts3D = pts3D.dot(coordChangeMat.T)

    projPts = pts3D.dot(camMat.T)
    projPts = np.stack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]],axis=1)

    assert len(projPts.shape) == 2

    return projPts

def showHandJoints(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''
    jointConns = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20], [0, 13, 14, 15, 16]]
    jointColsGt = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
    jointColsEst  = []
    for col in jointColsGt:
        newCol = (col[0]+col[1]+col[2])/3
        jointColsEst.append((newCol, newCol, newCol))
    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt[i], lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst[i], lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

def showObjJoints(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    '''
    Utility function for displaying object annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''
    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]
    jointColsGt = (255,255,0)
    newCol = (jointColsGt[0] + jointColsGt[1] + jointColsGt[2]) / 3
    jointColsEst  = (newCol, newCol, newCol)

    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt, lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst, lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-seq", "--sequence", required=True,
                    help="sequence name")
    ap.add_argument("-id", "--imageID", required=True,
                    help="image ID")
    args = vars(ap.parse_args())

    # if 'seq' not in args.keys():
    #     args['seq'] = 'MC1'
    #
    # if 'id' not in args.keys():
    #     args['id'] = '0025'


    id = args['id']
    seq = args['seq']

    # parse the annotation file for the sequence
    annoFilename = join(baseDir, 'sequences', seqName, 'anno.txt')
    annoDict = parseAnnoTxt(annoFilename)

    assert id in annoDict.keys(), 'File id %s not present in sequence %s.' % (id, seqName)

    # camera properties
    f = np.array([617.343, 617.343], dtype=np.float32)
    c = np.array([312.42, 241.42], dtype=np.float32)
    w = 640
    h = 480
    camMat = np.array([[f[0], 0., c[0]],[0., f[1], c[1]],[0., 0., 1.]])

    # get the annotations for the current id
    anno = annoDict[id]

    # read image, depths and object corner files
    imgFilename = join(baseDir, 'sequences', seqName, 'RGB', 'color_'+id+'.png')
    depthFilename = join(baseDir, 'sequences', seqName, 'Depth', 'depth_'+id+'.png')
    objCornersFilename = join(baseDir, 'models', anno['objID'], 'corners.npy')

    objCorners = np.load(objCornersFilename)
    img = cv2.imread(imgFilename)
    depth = decodeDepthImg(depthFilename)

    # transform the object corners
    objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

    # project 3D keypoints to image place
    handKps = project3DPoints(camMat, anno['handJoints'], isOpenGLCoords=True)
    objKps = project3DPoints(camMat, objCornersTrans, isOpenGLCoords=True)

    # show the 2D keypoints on image
    imgAnno = showHandJoints(img, handKps, lineThickness=2)
    imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)

    # visualize
    fig = plt.figure()
    ax = fig.subplots(2, 1)
    ax[0].imshow(imgAnno[:,:,[2,1,0]])
    ax[1].imshow(depth)
    plt.show()














