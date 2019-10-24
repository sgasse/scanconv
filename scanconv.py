import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2

from skimage import io, img_as_float, img_as_ubyte, transform
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_mean
from skimage.segmentation import active_contour, clear_border
from skimage.morphology import closing, square
from skimage.measure import label, regionprops, find_contours


def getImage():
#     orig = img_as_float(io.imread('img/scan_desk.jpg'))
    orig = img_as_float(io.imread('img/letter_sheared.jpg'))
    return orig


def preprocessImage(img):
    imgGray = rgb2gray(img)
    imgGray = gaussian(imgGray, sigma=1)
    return imgGray


def thresholdImage(imgGray):
    th = threshold_mean(imgGray)
    imgBinary = imgGray > th
    return imgBinary, th


def findRegion(imgBinary):
    bw = closing(imgBinary, square(3))
    cleared = clear_border(bw)

    labelImage = label(cleared)

    maxRegion = 0
    bbox = None
    for region in regionprops(labelImage):
        if region.area > maxRegion:
            bbox = region.bbox
            maxRegion = region.area
    return bbox


def farthestPoint(arr, quadrant):
    if quadrant == 1:
        # upper right
        quadArr = arr[(arr[:, 0] < 0) & (arr[:, 1] > 0)]
    elif quadrant == 2:
        # upper left
        quadArr = arr[(arr[:, 0] < 0) & (arr[:, 1] < 0)]
    elif quadrant == 3:
        # lower left
        quadArr = arr[(arr[:, 0] > 0) & (arr[:, 1] < 0)]
    elif quadrant == 4:
        # lower right 
        quadArr = arr[(arr[:, 0] > 0) & (arr[:, 1] > 0)]
    else:
        raise ValueError(f"There is no qudrant {quadrant}")
    dist = quadArr[:, 0]**2 + quadArr[:, 1]**2
    ind = np.argmax(dist)
    cornerPoint = quadArr[ind, :]
    return cornerPoint


def findPolygon(imgBinary):
    contourLength = 0
    contour = None
    for cont in find_contours(imgBinary.astype(np.float), 0.5):
        if len(cont) > contourLength:
            contourLength = len(cont)
            contour = cont

    rMax, cMax = imgBinary.shape
    centroid = np.array([rMax / 2, cMax / 2])
    relCont = contour - (centroid*np.ones((contour.shape)))

    # find contour point that is farthest away from the center per quadrant
    upperLeft = farthestPoint(relCont, 2) + centroid
    upperRight = farthestPoint(relCont, 1) + centroid
    lowerRight = farthestPoint(relCont, 4) + centroid
    lowerLeft = farthestPoint(relCont, 3) + centroid

    cropCont = np.array([upperLeft, lowerLeft, upperRight, lowerRight])

    width = int(max((upperRight[1] - upperLeft[1]),
                    (lowerRight[1] - lowerLeft[1])))
    height = int(max((lowerLeft[0] - upperLeft[0]),
                     (lowerRight[0] - upperRight[0])))

    return cropCont, [width, height]


def warpPerspective(img, cropCont, wh):
    width, height = wh
    width, height = [2100, 2970]
    src = cropCont[0:4, :].astype(np.float32)
    dst = np.array([[0, 0],
                    [height, 0],
                    [0, width],
                    [height, width]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    img = img_as_ubyte(img)
    imgWarped = cv2.warpPerspective(img, M, (width, height))

    return imgWarped

#     tf = transform.estimate_transform('projective', dst, src)
#     return transform.warp(img, tf, output_shape=(max(wh), max(wh)))


def plotImgs(orig, imgGray, imgBinary, bbox=None, cont=None, tr=None):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

    ax[0, 0].imshow(orig)
    ax[0, 0].set_title('Original')
    ax[0, 0].set_axis_off()

    ax[0, 1].imshow(imgGray, cmap=plt.cm.gray)
    ax[0, 1].set_title('Gray-Scale')
    ax[0, 1].set_axis_off()

    ax[1, 1].hist(imgGray.ravel(), bins=256)
    ax[1, 1].set_title('Histogram')

    ax[1, 0].imshow(imgBinary, cmap=plt.cm.gray)
    ax[1, 0].set_title('Thresholded')
    ax[1, 0].set_axis_off()

    if cont is not None:
        ax[1, 0].plot(cont[:, 1], cont[:, 0], '--r', linewidth=2)

    if bbox is not None:
        minr, minc, maxr, maxc = bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='blue', linewidth=2)
        ax[1, 0].add_patch(rect)

    if tr is not None:
        ax[0, 2].imshow(tr, cmap=plt.cm.gray)
        ax[0, 2].set_title('Transformed Image')
        ax[0, 2].set_axis_off()
    else:
        ax[0, 2].set_axis_off()

    ax[1, 2].set_axis_off()

    fig.tight_layout()

    plt.show()


orig = getImage()
imgGray = preprocessImage(orig)
imgBinary, th = thresholdImage(imgGray)
bbox = findRegion(imgBinary)
cropCont, wh = findPolygon(imgBinary)
transformedSheet = warpPerspective(imgGray, cropCont, wh)


plotImgs(orig, imgGray, imgBinary, bbox=bbox, cont=cropCont,
         tr=transformedSheet)

