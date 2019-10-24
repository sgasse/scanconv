import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_mean
from skimage.segmentation import active_contour, clear_border
from skimage.morphology import closing, square
from skimage.measure import label, regionprops


def getImage():
    orig = img_as_float(io.imread('img/scan_desk.jpg'))
    return orig


def preprocessImage(img):
    imgGray = rgb2gray(img)
    imgGray = gaussian(imgGray, sigma=1)
    return imgGray


def thresholdImage(imgGray):
    th = threshold_mean(imgGray)
    imgBinary = imgGray > th
#     imgBinary = imgBinary.astype(np.float)
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


def fitContour(imgBinary, relMargin=0.01):
    rMax, cMax = imgBinary.shape
    initRC = np.array([[relMargin*rMax, relMargin*cMax],
                       [relMargin*rMax, (1 - relMargin)*cMax],
                       [(1 - relMargin)*rMax, (1 - relMargin)*cMax],
                       [(1 - relMargin)*rMax, relMargin*cMax]])
    snake = active_contour(imgBinary, initRC,
                           alpha=0.1, beta=0.001, w_line=0, gamma=0.01,
                           max_px_move=4.0, max_iterations=300,
                           convergence=0.1)
    return [initRC, snake]


def plotImgs(orig, imgGray, imgBinary, bbox=None):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))

    ax[0, 0].imshow(orig)
    ax[0, 0].set_title('Original')
    ax[0, 0].set_axis_off()

    ax[1, 0].imshow(imgGray, cmap=plt.cm.gray)
    ax[1, 0].set_title('Gray-Scale')
    ax[1, 0].set_axis_off()

    ax[1, 1].hist(imgGray.ravel(), bins=256)
    ax[1, 1].set_title('Histogram')

    ax[2, 0].imshow(imgBinary, cmap=plt.cm.gray)
    ax[2, 0].set_title('Thresholded')
    ax[2, 0].set_axis_off()

    if bbox is not None:
        minr, minc, maxr, maxc = bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='blue', linewidth=2)
        ax[2, 0].add_patch(rect)

    fig.tight_layout()

    plt.show()


orig = getImage()
imgGray = preprocessImage(orig)
imgBinary, th = thresholdImage(imgGray)
bbox = findRegion(imgBinary)


plotImgs(orig, imgGray, imgBinary, bbox=bbox)

