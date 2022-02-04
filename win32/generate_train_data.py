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
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cv2
import shutil
import shapely.geometry as sp

def GetCmtCategory(filename) :
    f1 = open(filename, 'r')
    lines = f1.readlines()
    category = 'N'
    cmtgroup = "unknown"
    for line in lines :
        if (line.startswith("#")) :
            tokens = line.split(":")
            if (len(tokens) == 2 and "Brightness Category" in tokens[0]):
                category = tokens[1].strip()
            elif (len(tokens) == 2 and "Comet Group" in tokens[0]):
                cmtgroup = tokens[1].strip()

    return category, cmtgroup

#from google.colab.patches import cv2_imshow
def GenerateCometSequencesData(trainPath, truthfile, enableCategory) :
  data_set = []
  with open(truthfile, 'r') as f:
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
            paths.append(os.path.join(trainPath, tokens[0], tokens[1+i*3]))
            truths[tokens[1+i*3]] = [float(tokens[2+i*3]),tokens[3+i*3]]
        images.sort()
        paths.sort()
        seq["images"] = images
        seq["path"] = paths
        seq["truth"] = truths
        if len(images)>0:
            data_set.append(seq)

    if (enableCategory) :
        for seq in data_set:
            filepath = trainPath + "\\" + seq["ID"] + "\\" + seq["ID"] + ".txt"
            category,group = GetCmtCategory(filepath)
            seq["category"] = category
            seq["group"] = group

  return data_set

def drawCometSeq(img, cmtseq) :
  #index = 0
  retval = True
  blobs = cmtseq["pos"]
  if (len(blobs) < 5):
      print("Rejected---------------------------------")
      retval = False

  for i in range(1,len(blobs)) :
    y2, x2, r1 = blobs[i]
    y1, x1, r2 = blobs[i-1]
    #print("KK", x1,"-",y1,"---",x2,"-", y2)
    # note distance
    global train_dist
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    #dist = int(dist/100)
    #train_dist[dist] = train_dist[dist] + 1 if dist in train_dist else 1

    #note angle
    if (i > 1) :
        global train_angle
        y0, x0, r0 = blobs[i-2]
        v1 = [y2-y1, x2-x1]
        v2 = [y0-y1, x0-x1]
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(v1, v2)
        if ((dot_product <= -1.0 or dot_product >= 1.0) and dist > 60):
            retval = False
            print("Rejected---------------------------------")
            #break
        angle = np.arccos(dot_product)
        if ((angle < np.pi - 0.3 or angle > np.pi + 0.3) and dist > 60):
            #x = 1
            retval = False
            print("Rejected---------------------------------", angle)
            #break
            #angle = int((np.pi-angle)/100)
            #train_angle[angle] = train_angle[angle] + 1 if angle in train_angle else 1

    cv2.line(img, (int(float(x1)),int(float(y1))), (int(float(x2)),int(float(y2))), (255,255,255))
    cv2.circle(img, (int(x1),int(y1)), 4, (255,255,255), cv2.FILLED)
  return retval

def processSeq1(seq, outpath) :
    print("Processing: " + seq["ID"])
    numImg = len(seq["path"])
    cmtseq = []
    for i in range(numImg):
        imgName = seq["images"][i]
        truthPoint = seq["truth"][imgName]
        cmtseq.append((float(truthPoint[1]), float(truthPoint[0]), 1))

    blank_image = np.zeros((1024, 1024, 3), np.uint8)
    cmtdict = {}
    cmtdict["ID"] = seq["ID"]
    cmtdict["pos"] = cmtseq
    cmtdict["dist"] = []
    cmtdict["angle"] = []
    cmtdict["score"] = []
    cmtdict["speed"] = []
    # Process dist vector
    blobs = cmtdict["pos"]
    for i in range(1,len(blobs)) :
        y2, x2, r1 = blobs[i]
        y1, x1, r2 = blobs[i-1]
        global train_dist
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        cmtdict["dist"].append(round(dist,2))
        dist = int(dist/100)
        global train_dist
        train_dist[dist] = train_dist[dist] + 1 if dist in train_dist else 1
        
        # Process Angle vector
        if (i > 1) :
            y0, x0, r0 = blobs[i-2]
            v1 = [y2-y1, x2-x1]
            v2 = [y0-y1, x0-x1]
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            dot_product = np.dot(v1, v2)
            if dot_product > -1.0 or dot_product < 1.0:
                angle = np.arccos(dot_product)
                cmtdict["angle"].append(round(math.degrees(abs(np.pi-angle)), 2))
                angle = int(abs(np.pi-angle)/100)
                global train_angle
                train_angle[angle] = train_angle[angle] + 1 if angle in train_angle else 1

    # Process time vector
    data_cube = np.empty((1024, 1024,numImg))

    timestamps = [0] * numImg
    cmtdict["time"] = []
    for i in range(numImg):
        img, hdr = fits.getdata(seq["path"][i], header=True)
        img = img.astype('float64') / hdr['EXPTIME']
        timestamps[i] = hdr["MID_TIME"]
        # ndimage.median_filter(img, size=9)
        data_cube[:,:,i] = img - ndimage.median_filter(img, size=9) #medfilt2d(img, kernel_size=9)
        rdiff = np.diff(data_cube, axis=2)

        if (i >= 1) :
            cmtdict["time"].append(round(timestamps[i] - timestamps[i-1], 2))

    # Process speed vector
    #print("Dist", len(cmtdict["dist"]))
    #print("Time", len(cmtdict["time"]))
    #for idx in range(0, len(cmtdict["dist"] - 1)):
    #    cmtdict["speed"].append(round(cmtdict["dist"][i] / cmtdict["time"][i], 4))
    for (item1, item2) in zip(cmtdict["dist"], cmtdict["time"]):
        cmtdict["speed"].append(round(item1/item2,4))
    
    #Process the images of the sequence and apply transformation, then capture the score (10 -> point exist, 8 -> within 10px, 4 -> winthin 25px, else 0)
    imglist = []
    for i in range(numImg-1):
      print("Processing Img Score" + str(i))
      medsub = -rdiff[:,:,i]
      medsub = cv2.min(medsub, 10.)
      medsub = cv2.max(medsub, -10.)
      medsub = (medsub + 10.) * 255. / 20.
      medsub = np.uint8(medsub)
      (T, mask) = cv2.threshold(medsub, 190, 255, cv2.THRESH_BINARY)
      medsub = cv2.bitwise_and(medsub, mask)
      cv2.circle(medsub, (512,512), 190, (0,0,0), cv2.FILLED)
      imglist.append(medsub)

      blobs_log = blob_log(medsub, max_sigma=5, min_sigma=1, num_sigma=5, threshold=0.1)
      if len(blobs_log) > 0:
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

      # Calculate triuth zone
      #found = 0 # 0 means nothing found near truth point, 10 means found within radius 10px, 2 means found within radius 25px
      yc,xc,rc = cmtdict["pos"][i]
      truthzone10 = sp.Point(float(xc), float(yc)).buffer(10)
      truthzone25 = sp.Point(float(xc), float(yc)).buffer(25)
      score = 0
      if (medsub[int(xc)][int(yc)] != 0.0):
          score = 10
      else :
        for blob in blobs_log:
            y, x, r = blob # note blob stores in cloumn order
            if (truthzone10.contains(sp.Point(x, y))) :
                score = 8
                break

            if (truthzone25.contains(sp.Point(x, y))) :
                score = 4
                break

      cmtdict["score"].append(score)

    drawCometSeq(blank_image, cmtdict)
    cv2.imwrite(outpath + "\\" + seq["ID"]+".png", blank_image)
    return cmtdict

def write_to_file(cmtdict, filename) :
    f1 = open(filename, 'a+')

    f1.write('%s :\n' % (cmtdict["ID"]))
    # write angle vector
    f1.write("Angle: [")
    for val in cmtdict["angle"]: 
        f1.write('%s, ' % (val))
    f1.write("]\n")

    # write time vector
    f1.write("Time: [")
    for val in cmtdict["time"]: 
        f1.write('%s, ' % (val))
    f1.write("]\n")

    # write Distance vector
    f1.write("Dist: [")
    for val in cmtdict["dist"]: 
        f1.write('%s, ' % (val))
    f1.write("]\n")

    # write speed vector
    f1.write("Speed: [")
    for val in cmtdict["speed"]: 
        f1.write('%s, ' % (val))
    f1.write("]\n")

    # write score vector
    f1.write("Speed: [")
    for val in cmtdict["score"]: 
        f1.write('%s, ' % (val))
    f1.write("]\n")

def processSeq(seq, outdir) :
    print("Processing: " + seq["ID"] + " group: " + seq["group"])
    numImg = len(seq["path"])
    cmtseq = []
    outpath = outdir # + "\\" + seq["group"] (Enable it if you need to generate by category)
    if (not os.path.exists(outpath)):
        os.mkdir(outpath, 777)

    for i in range(numImg):
        imgName = seq["images"][i]
        truthPoint = seq["truth"][imgName]
        cmtseq.append((float(truthPoint[1]), float(truthPoint[0]), 1))

    blank_image = np.zeros((1024, 1024, 3), np.uint8)
    cmtdict = {}
    cmtdict["pos"] = cmtseq
    retval = drawCometSeq(blank_image, cmtdict)
    if (retval) :
        cv2.imwrite(outpath + "\\" + seq["ID"]+".png", blank_image)
    else :
        rejectpath = outpath + "\\rejected"
        if (not os.path.exists(rejectpath)):
            os.mkdir(rejectpath, 777)
        cv2.imwrite(rejectpath + "\\" + seq["ID"]+".png", blank_image)

    return cmtdict

#path = "/content/drive/MyDrive/Comet/train1/train"
#truthfile = "/content/drive/MyDrive/Comet/train1/train-gt.txt"
#truthfile = "/content/drive/MyDrive/Comet/train1/

train_dist  = { }
train_angle = { }

folder_in = sys.argv[1] + "\\train"
truth_filename = sys.argv[1] + "\\train-gt.txt"
train_dir = sys.argv[1] + "\\comet_normalized"
train_info_file = sys.argv[1] + "\\train-info.txt"
cmt_info_file = sys.argv[1] + "\\cmt-info.txt"

if (os.path.exists(train_dir)):
    shutil.rmtree(train_dir)

os.mkdir(train_dir, 777)

data_set = GenerateCometSequencesData(folder_in, truth_filename, True)
for seq in data_set :
    cmtdict = processSeq(seq, train_dir)
    #write_to_file(cmtdict, cmt_info_file)

# Log train info in file
sortedangle = sorted(train_angle.items())
sorteddist = sorted(train_dist.items())
f1 = open(train_info_file, 'w')
f1.write("Angle\n")
#f1.write(train_angle)
for key, value in sortedangle: 
    f1.write('%s:%s\n' % (key, value))
f1.write("Distance\n")
for key, value in sorteddist: 
    f1.write('%s:%s\n' % (key, value))
#f1.write(train_dist)
f1.close()




# path = "C:\\github\\nasa-comet\\train1\\train\\cmt0006\\cmt0006.txt"
# c,g = GetCmtCategory(path)
# print(c)
# print(g)