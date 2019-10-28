import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import io, img_as_float, img_as_ubyte, transform
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_mean
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.measure import label, regionprops, find_contours
from PIL import Image
from PyPDF2 import PdfFileReader, PdfFileWriter
from shutil import rmtree


def catPDF(inFiles, outFile):
    inStream = list()
    try:
        for file in inFiles:
            inStream.append(open(file, 'rb')) 
        writer = PdfFileWriter()
        for reader in map(PdfFileReader, inStream):
            for n in range(reader.getNumPages()):
                writer.addPage(reader.getPage(n))
        writer.write(outFile)
    finally:
        for f in inStream:
            f.close()


def getImage(file):
    orig = img_as_float(io.imread(file))
    imgGray = gaussian(rgb2gray(orig), sigma=1)
    return orig, imgGray


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

    cropCont = np.array([upperLeft, lowerLeft, lowerRight, upperRight,
                         upperLeft])

    return cropCont


def warpPerspective(img, cropCont):
    if img.shape[0] > img.shape[1]:
        # portrait
        width, height = [2100, 2970]
    else:
        # landscape
        width, height = [2970, 2100]

    src = cropCont[0:4, :]
    dst = np.array([[0, 0],
                    [height, 0],
                    [height, width],
                    [0, width]], dtype=np.float32)

    tf = transform.estimate_transform('projective',
                                      np.flip(dst, 1),
                                      np.flip(src, 1))
    return transform.warp(img, tf, output_shape=(height, width))


def savePDF(img, filename, quality=80):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        imgP = Image.fromarray(img_as_ubyte(img))
    imgP.save(filename, 'pdf')


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


def processImage(fullname):
    orig, imgGray = getImage(fullname)
    imgBinary, _ = thresholdImage(imgGray)
    cropCont = findPolygon(imgBinary)
    imgWarped = warpPerspective(orig, cropCont)
    return imgWarped


def batchTransform(imgDir):
    tmpDir = '/tmp/scanconv'
    if os.path.exists(tmpDir):
        rmtree(tmpDir)
    os.makedirs(tmpDir)
    pdfDir = 'pdfs'
    os.makedirs(pdfDir, exist_ok=True)
    pdfDict = dict()
    for root, _, files in os.walk(imgDir):
        # convert files and create separate PDFs
        for file in files:
            if file.endswith('.jpg') or file.endswith('.JPG'):
                origFile = os.path.join(root, file)
                if root == imgDir:
                    pdfFile = os.path.join(pdfDir, (file.rsplit('.')[0] + '.pdf'))
                else:
                    pdfFile = os.path.join(tmpDir, (file.rsplit('.')[0] + '.pdf'))
                    if root not in pdfDict:
                        pdfDict[root] = [pdfFile]
                    else:
                        pdfDict[root].append(pdfFile)
                imgWarped = processImage(origFile)
                savePDF(imgWarped, pdfFile)

    for docName in pdfDict.keys():
        inFiles = pdfDict[docName]
        with open(os.path.join(pdfDir, f'{docName}.pdf'), 'wb') as outFile:
            catPDF(inFiles, outFile)

    rmtree(tmpDir)


def test_imgConv():
    orig, imgGray = getImage('img/letter_sheared.jpg')
    imgBinary, th = thresholdImage(imgGray)
    cropCont = findPolygon(imgBinary)
    imgWarped = warpPerspective(orig, cropCont)

    os.makedirs('test_output', exist_ok=True)
    savePDF(imgWarped, 'test_output/letter_done.pdf')

    plotImgs(orig, imgGray, imgBinary, bbox=None, cont=cropCont,
             tr=imgWarped)

    resDict = {'orig': orig,
               'imgGray': imgGray,
               'imgBinary': imgBinary,
               'cropCont': cropCont,
               'imgWarped': imgWarped}

    return resDict


if __name__ == '__main__':
    print('Scan Converter')
    if len(sys.argv) == 1:
        path = os.path.abspath('.')
    elif len(sys.argv) == 2:
        path = os.path.abspath(f'./{sys.argv[1]}')
        if not os.path.exists(path):
            print(f'Given path {path} does not exist')
            sys.exit()
    else:
        print('Arguments not understood')
        sys.exit()

    print(f'Parsing {path}')
    batchTransform(path)