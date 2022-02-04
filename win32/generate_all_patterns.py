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
from generate_train_data import GenerateCometSequencesData
#from google.colab.patches import cv2_imshow

DEBUG = 1

def Log(msg) :
  if DEBUG :
    print(msg)

# def GenerateCometSequencesData(trainPath, truthfile) :
#   data_set = []
#   with open(truthfile, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         tokens = line.split(',')
#         seq = {}
#         seq["ID"] = tokens[0]
#         images = []
#         paths = []
#         truths = {}
#         for i in range( (len(tokens)-2)//3 ):
#             images.append(tokens[1+i*3])
#             paths.append(os.path.join(trainPath, tokens[0], tokens[1+i*3]))
#             truths[tokens[1+i*3]] = [float(tokens[2+i*3]),tokens[3+i*3]]
#         images.sort()
#         paths.sort()
#         seq["images"] = images
#         seq["path"] = paths
#         seq["truth"] = truths
#         if len(images)>0:
#             data_set.append(seq)
#   return data_set

def IncludeInLine(p1, p2, p3) :
  x1, y1 = p1
  x2, y2 = p2
  x3, y3 = p3
  v1 = [y3-y2, x3-x2]
  v2 = [y1-y2, x1-x2]
  v1 = v1 / np.linalg.norm(v1)
  v2 = v2 / np.linalg.norm(v2)
  dot_product = np.dot(v1, v2)
  if dot_product<=-1.0 or dot_product>=1.0:
    return False  
  angle = np.arccos(dot_product)
  if angle<np.pi-0.2 or angle>np.pi+0.2: # match blobs that forms a near straight line
    return False
  return True

def PreProcessImageSequence(seq) :
  numImg = len(seq["path"]) 
  print("Processing for all patterns: ", seq["ID"])
  width = 1024
  height = 1024

  # Create 3D data cube to hold data, assuming all data have
  # array sizes of 1024x1024 pixels.
  data_cube = np.empty((width,height,numImg))

  #timestamps = [0] * numImg
  #Log(len(timestamps))
  for i in range(numImg):
    # read image and header from FITS file
    img, hdr = fits.getdata(seq["path"][i], header=True)
        
    # Normalize by exposure time (a good practice for LASCO data)
    img = img.astype('float64') / hdr['EXPTIME']
        
    # Collect timestamps
    #timestamps[i] = hdr["MID_TIME"]
        
    # Store array into datacube (3D array)
    # ndimage.median_filter(img, size=9)
    data_cube[:,:,i] = img - ndimage.median_filter(img, size=9) #medfilt2d(img, kernel_size=9)
    
    #timestamps.append(0)
    #seq["time"] = timestamps
    # Take the difference between consecutive images
    rdiff = np.diff(data_cube, axis=2)
    
  imglist = []
  for i in range(numImg-1):
      Log("Processing Img " + str(i))
      medsub = -rdiff[:,:,i]
      medsub = cv2.min(medsub, 10.)
      medsub = cv2.max(medsub, -10.)
      medsub = (medsub + 10.) * 255. / 20.
      medsub = np.uint8(medsub)
      (T, mask) = cv2.threshold(medsub, 190, 255, cv2.THRESH_BINARY)
      medsub = cv2.bitwise_and(medsub, mask)
      # mask out sun
      cv2.circle(medsub, (512,512), 190, (0,0,0), cv2.FILLED)

      # search for blobs
      #blobs_log = blob_log(medsub, max_sigma=5, min_sigma=1, num_sigma=5, threshold=0.1)
      #if len(blobs_log)>0:
      #    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
      #comets.append(blobs_log)
      imglist.append(medsub)
  return imglist

def PopulateCometLikeSeq(img, cmtseqlist, itr) :
  # if the list is empty, add all the blobs
  # currseqid = cmtseqlist["itreation"]
  if (itr == 0) :
    blobs_log = blob_log(img, max_sigma=5, min_sigma=1, num_sigma=5, threshold=0.1)
    if len(blobs_log) > 0:
      blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
     
    #seq = []
    #seq.append(currseqid)
    for blob in blobs_log :
      cmt = { }
      cmt["seq"] = [0]
      cmt["pos"] = [blob]
      cmt["timeseq"] = 0
      cmt["score"] = 0 # 1 if exact pos matches, 0.9 is within 10px radius, 0.6 is within 25px radius, 0 otherwise
      #print(cmt["pos"])
      cmtseqlist.append(cmt)
    return cmtseqlist

  if (itr == 1) :
    blobs_log = blob_log(img, max_sigma=5, min_sigma=1, num_sigma=5, threshold=0.1)
    if len(blobs_log) > 0:
      blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
     
      cmtseqlist1 = []
      for cmtseq in cmtseqlist :
        p1y, p1x, r1 = cmtseq["pos"][0]
        #print(cmtseq["pos"])
        for blob in blobs_log :
          p2y, p2x, r2 = blob
          dist = np.sqrt((p2x - p1x)**2 + (p2y - p1y)**2)
          if (dist <= 100 and dist >= 5) :
            cmt = { }
            cmt["seq"] = [0, 1]
            p1 = p1y, p1x, r1
            p2 = p2y, p2x, r2
            cmt["pos"] = [p1, p2]
            cmt["timeseq"] = 0
            cmt["score"] = 1
            cmtseqlist1.append(cmt)
            #break
    return cmtseqlist1
  else :
    blobs_log = blob_log(img, max_sigma=5, min_sigma=1, num_sigma=5, threshold=0.1)
    if len(blobs_log) > 0:
      blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    
    for cmtseq in cmtseqlist :
      y1, x1, r1 = cmtseq["pos"][-2]
      y2, x2, r2 = cmtseq["pos"][-1]
      for blob in blobs_log:
        y3, x3, r3 = blob
        dist = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        if (dist <= 100 and dist >= 10):
          finclude = IncludeInLine((x1,y1), (x2,y2), (x3,y3))
          if (finclude) :
            cmtseq["pos"].append((y3, x3, r3))
            break # consider taking all combinations
        #else :
        #  print("XXX")
    return cmtseqlist

def drawCometLikeSeq(img, cmtseq) :
  #index = 0
  blobs = cmtseq["pos"]
  for i in range(1,len(blobs)) :
    y2, x2, r1 = blobs[i]
    y1, x1, r2 = blobs[i-1]
    #print("KK", x1,"-",y1,"---",x2,"-", y2)
    cv2.line(img, (int(float(x1)),int(float(y1))), (int(float(x2)),int(float(y2))), (255,255,255))
    cv2.circle(img, (int(x1),int(y1)), 4, (255,255,255), cv2.FILLED)

# Consider running drawCometSeq() from generate train data
def DrawCometSeqPatterns(path, seq, cmtseqlist) :
    outpath = path + "\\" + seq["ID"]
    if (not os.path.exists(outpath)):
        os.mkdir(outpath, 777)

    #blank_image = np.zeros((1024, 1024, 3), np.uint8)
    #print("list-len:", len(cmtseqlist))
    cnt = 1
    for cmtseq in cmtseqlist :
        #if (len(cmtseq["pos"]) >= (len(seq["path"]) - 2)):
        #print("Processing comet sequence: " + seq["ID"] + ": " + str(cnt))
        blank_image = np.zeros((1024, 1024, 3), np.uint8)
        drawCometLikeSeq(blank_image, cmtseq)
        cv2.imwrite(outpath + "\\" + seq["ID"]+ "_" + str(cnt) + ".png", blank_image)
        cnt = cnt + 1
    
    #print("actual-list-len:", cnt)

# Checks if two cmt like sequences are near duplicates
# return 0 -> They are not duplicate
# return 1 -> they are duplicate and select argument 1
# return 2 -> they are duplicate and select 
def IsNearDuplicate1(cmtseq1, cmtseq2) :
  cmtseqT1 = cmtseq1["pos"].copy()
  cmtseqT2 = cmtseq2["pos"].copy()
  cmtseqT1.sort(key = lambda a: a[0])
  cmtseqT2.sort(key = lambda a: a[0])
  #print(cmtseqT1)
  #print(cmtseqT2)
  nearcount = 0
  farcount = 0
  minlen = min(len(cmtseqT1), len(cmtseqT2))
  #print(minlen)
  #del cmtseqT1[minlen :]
  #del cmtseqT2[minlen :]
  for t1, t2 in zip(cmtseqT1, cmtseqT2) :
      y1, x1, r1 = t1
      y2, x2, r2 = t2
      dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
      if (dist < 30) :
        nearcount = nearcount + 1
      else:
        farcount = farcount + 1

      if (farcount >= 2) :
        return 0

  # we are here means, they are near duplicate
  if (len(cmtseq1["pos"]) >= len(cmtseq2["pos"])) :
    return 1
  elif(len(cmtseq2["pos"]) > len(cmtseq1["pos"])):
    return 2
  else :
    raise Exception("Invalid cmt sequence truncation")

# Cleans up the cmt sequence list by removing very small sequences (< 5) and removing near duplicate dequences
def CleanUpCmtSeqList(cmtseqlist, seq):
  cmtseqlist1 = []
  for cmtseq in cmtseqlist:
    if (len(cmtseq["pos"]) >= 5 and len(cmtseq["pos"]) <= len(seq["path"]) - 2):
      cmtseq["mark"] = False
      cmtseqlist1.append(cmtseq)

  cmtseqlist2 = []
  for i in range(0,len(cmtseqlist1)) :
    cmtseq1 = cmtseqlist1[i]
    if (cmtseq1["mark"] == True):
      continue

    cmt = { }
    for j in range(i+1, len(cmtseqlist1)) :
        cmtseq2 = cmtseqlist1[j]
        if (cmtseq2["mark"] == True):
          continue

        val = IsNearDuplicate1(cmtseq1, cmtseq2)
        if (val == 1):
          cmtseq1["mark"] = True
          cmtseq2["mark"] = True # means considered in duplicate evaluation 
          cmt = cmtseq1
        elif(val == 2) :
          cmtseq1["mark"] = True
          cmtseq2["mark"] = True 
          cmt = cmtseq2
        else:
          cmt = cmtseq1
    if (cmt):
        cmtseqlist2.append(cmt)

  return cmtseqlist2


folder_in = sys.argv[1] + "\\train"
truth_filename = sys.argv[1] + "\\train-gt.txt"
train_dir = sys.argv[1] + "\\noncomet"
#train_info_file = sys.argv[1] + "\\train-info.txt"
#cmt_info_file = sys.argv[1] + "\\cmt-info.txt"

if (os.path.exists(train_dir)):
    shutil.rmtree(train_dir)

os.mkdir(train_dir, 777)

data_set = GenerateCometSequencesData(folder_in, truth_filename, True)
for seq in data_set :
    # NOTE : It takes time, take small data set first
    print("Generating comet like sequences for :", seq["ID"])
    imglist = PreProcessImageSequence(seq)
    index = 0
    cmtseqlist = []
    for img in imglist :
        cmtseqlist = PopulateCometLikeSeq(img, cmtseqlist, index)
        index = index + 1

    # TODO : Remove near duplicate commet sequences and also ensure the actual comet sequence is eliminated
    cmtseqlist1 = CleanUpCmtSeqList(cmtseqlist, seq)

    # After processing all images, the cmtseqlist will have all cmt like patterns
    # Draw each pattern in the out put path in separate images
    DrawCometSeqPatterns(train_dir, seq, cmtseqlist1)


