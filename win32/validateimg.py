import sys
import pickle
import os
import math
import random
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt2d
from scipy.spatial import distance
import matplotlib.pyplot as plt
import cv2
import shutil
import shapely.geometry as sp

THRESHOLD_LIMIT = 160 # this value determines how much noise (blobs) we want to expose

min_dist = 1024
max_dist = 0

def process_sequence(seq):
    print("Processing: " + seq["ID"])
    numImg = len(seq["path"])
    seq["T10"] = 0
    seq["T25"] = 0
    seq["status"] = 0
    prevPoint = [0,0]
    for i in range(numImg):
        imgName = seq["images"][i]
        truthPoint = seq["truth"][imgName]
        #print("imgName: ", imgName, "[X,Y]: ", float(truthPoint[1]))
        # read image and header from FITS file
        img, hdr = fits.getdata(seq["path"][i], header=True)
        
        # Normalize by exposure time (a good practice for LASCO data)
        img = img.astype('float64') / hdr['EXPTIME']
        img1 = medfilt2d(img, kernel_size=9)
        img2 = img - img1
        img2 = cv2.min(img2, 10)
        img2 = cv2.max(img2, -10)
        img2 = (img2 + 10.) * 255. / 20.
        #img2 = np.uint8(img2)
        (T, mask) = cv2.threshold(img2, THRESHOLD_LIMIT, 255, cv2.THRESH_BINARY)
        img2 = cv2.bitwise_and(img2, mask)
        cv2.circle(img2, (512,512), 190, (0,0,0), cv2.FILLED)
        #cv2.imwrite(visualize_dir + "\\" + seq["ID"] +"_"+str(i)+".png", img2)

        # Get all the blobs from processed image
        blobs_log = blob_log(img2, max_sigma=5, min_sigma=1, num_sigma=5, threshold=0.1)
        if len(blobs_log) > 0:
            blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
        
        # check if truthzone 10pixel is found
        found = 0 # 0 means nothing found near truth point, 10 means found within radius 10px, 2 means found within radius 25px
        truthzone10 = sp.Point(float(truthPoint[0]), float(truthPoint[1])).buffer(10)
        truthzone25 = sp.Point(float(truthPoint[0]), float(truthPoint[1])).buffer(25)
        currpoint = [float(i) for i in truthPoint]
        cv2.circle(img2, (int(currpoint[0]),int(currpoint[1])), 5, (255,255,255), cv2.FILLED)
        print("Truth: ",i,": ", tuple(truthPoint))
        for blob in blobs_log:
            y, x, r = blob # note blob stores in cloumn order
            if (truthzone10.contains(sp.Point(x, y))) :
                cv2.circle(img2, (int(x), int(y)), 7, 150, 1)
                seq["T10"]  = seq["T10"]  + 1
                seq["T25"]  = seq["T25"]  + 1
                print("Actual10: ", (x, y))
                break

            if (truthzone25.contains(sp.Point(x, y))) :
                print("Actual25: ", (x, y))
                cv2.circle(img2, (int(x), int(y)), 7, 255, 1)
                seq["T25"]  = seq["T25"]  + 1
                break

        cv2.imwrite(visualize_dir + "\\" + seq["ID"] +"_"+str(i)+".png", img2)

        # Update the min and max distance
        currpoint = [float(i) for i in truthPoint]
        global max_dist
        global min_dist
        if (tuple(prevPoint) != (0, 0)) :
            max_dist = max(math.dist(tuple(currpoint), tuple(prevPoint)), max_dist)
            min_dist = min(math.dist(tuple(currpoint), tuple(prevPoint)), min_dist)
        
        prevPoint = currpoint

    # Update result in seq
    seq["status"] = seq["T10"] >= 5 and seq["T25"] >= int((numImg + 5)/2)
    #print (seq["ID"], " : ",seq["status"], "[T10 :", seq["T10"], "/", 5, ", ", "T25 :", seq["T25"], "/", int((numImg + 5)/2))
    return seq

# folder for the set
folder_in = sys.argv[1]
# ground truth file to be visualized
comet_filename = sys.argv[2] + "\\train-gt.txt"
visualize_dir = sys.argv[2] + "\\validate"
validate_filename = sys.argv[2] + "\\validateresults.txt"

data_set = []
with open(comet_filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        tokens = line.split(',')
        seq = {}
        seq["ID"] = tokens[0]
        images = []
        paths = []
        truths = {}
        for i in range( (len(tokens)-2)//3 ):
            images.append(tokens[1+i*3])
            paths.append(os.path.join(folder_in, tokens[0], tokens[1+i*3]))
            truths[tokens[1+i*3]] = [float(tokens[2+i*3]),tokens[3+i*3]]
        images.sort()
        paths.sort()
        seq["images"] = images
        seq["path"] = paths
        seq["truth"] = truths
        if len(images)>0:
            data_set.append(seq)

if (os.path.exists(visualize_dir)):
    shutil.rmtree(visualize_dir)

os.mkdir(visualize_dir, 777)
outstr = []
trueCount = 0
for s in data_set:
    seq1 = process_sequence(s)
    outstr.append(seq1["ID"] + " : " + str(seq1["status"]) + " [T10 :" + str(seq1["T10"]) + "/5" + ", " + "T25 :" + str(seq1["T25"]) + "/" + str(int((len(seq1["path"]) + 5)/2)) + "].\n")
    if (seq1["status"]) :
        trueCount = trueCount + 1

print("MIN: ", min_dist, " MAX: ", max_dist)
f1 = open(validate_filename, 'w')
for st in outstr :
    print(st)
    f1.write(st)

print("TrueCount : " + str(trueCount) + "/" + str(len(outstr)))
f1.write("TrueCount : " + str(trueCount) + "/" + str(len(outstr)) + "\n")
f1.write("MIN_DIST :" + str(min_dist) + "\n")
f1.write("MAX_DIST :" + str(max_dist) + "\n")
f1.close()