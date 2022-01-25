assert __name__ == "__main__"

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
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import faulthandler; faulthandler.enable()
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# Enable to see debug info
DEBUG = False

def search_comet(seq, comets, idx, x, y, idata, dist, timestamp):
    ''' Recursively search through the frames for a comet.
        Trying to link up a path with a newly detected blob in the next frames.
        Tracks a blob based on the expected frame location given the time for each frame and calculated speed based on previous frames.
    '''
    cometsFound = []
    # going past the length of the sequence
    if idx>=len(comets):
        return cometsFound

    # First frame, add all blobs
    if len(x)==0:
        for blob in comets[idx]:
            py, px, pr = blob
            x.append(px)
            y.append(py)
            timestamp.append(seq["time"][idx])
            idata.append(idx)
            cometsFound += search_comet(seq, comets, idx+1, x, y, idata, dist, timestamp)
            x.pop()
            y.pop()
            timestamp.pop()
            idata.pop()
            if len(cometsFound)>20: # Early exit when we detected 20+ comets
                break
        return cometsFound

    # Tracking one blob, trying to add another within range
    if len(x)==1:
        for blob in comets[idx]:
            py, px, pr = blob
            # Quick check for the distance from the previous blob, only match nearby blobs
            if abs(py-y[-1])>40 or abs(px-x[-1])>40: 
                continue
            sqdist = (py-y[-1])**2 + (px-x[-1])**2
            if sqdist<25: # must move at least 5 pixels
                continue
            x.append(px)
            y.append(py)
            idata.append(idx)
            timestamp.append(seq["time"][idx])
            cometsFound += search_comet(seq, comets, idx+1, x, y, idata, np.sqrt(sqdist), timestamp)
            x.pop()
            y.pop()
            timestamp.pop()
            idata.pop()
            if len(cometsFound)>20: # Early exit when we detected 20+ comets
                break
        return cometsFound

    # Tracking two blobs, trying to add another within range and direction of travel. Then check the number of blobs on the extrapolated path
    if len(x)==2:
        
        for blob in comets[idx]:
            py, px, pr = blob
            # Quick check for the distance from the previous blob
            if abs(py-y[-1])>40 or abs(px-x[-1])>40: 
                continue
            # calculate expected distance of next blob
            sqrdist = np.sqrt((py-y[-1])**2 + (px-x[-1])**2)
            expdist = (seq["time"][idx] - timestamp[-1]) * dist / (timestamp[-1]-timestamp[0])
            mindist = 0.8 * expdist
            maxdist = 1.2 * expdist
            # Only allow matches within similar distances
            if sqrdist<10 or sqrdist<mindist or sqrdist>maxdist:
                continue
            # check angle
            v1 = [py-y[-1], px-x[-1]]
            v2 = [y[0]-y[-1], x[0]-x[-1]]
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            dot_product = np.dot(v1, v2)
            if dot_product<=-1.0 or dot_product>=1.0:
                continue
            angle = np.arccos(dot_product)
            if angle<np.pi-0.3 or angle>np.pi+0.3: # match blobs that forms a near straight line
                continue
            x.append(px)
            y.append(py)
            idata.append(idx)
            timestamp.append(seq["time"][idx])
                        
            # Extrapolate ahead
            # Start position and time
            sx = x[0]
            sy = y[0]
            stime = seq["time"][idata[0]]
            # Direction and speed
            vx = x[-1] - sx
            vy = y[-1] - sy
            vtime = seq["time"][idata[-1]] - stime
            matches = 0 
            for a in range(1,4):  # look a few steps ahead
                if idx+a<len(comets):
                    nidx = idx+a
                    dtime = seq["time"][nidx] - stime
                    # The comet should roughly be at xx,yy in this next frame
                    xx = sx + vx*dtime/vtime
                    yy = sy + vy*dtime/vtime
                    foundOne = False
                    if xx>=0 and xx<1024 and yy>=0 and yy<1024:
                        # Try to find a blob along this line
                        for nextblob in comets[nidx]:
                            ny, nx, nr = nextblob
                            if abs(yy-ny)>4 or abs(xx-nx)>4: # Restrict to very close matches
                                continue
                            matches += 1
                            x.append(nx)
                            y.append(ny)
                            idata.append(nidx)
                            timestamp.append(seq["time"][nidx])
                            foundOne = True
                            break
                    if a>1 and not foundOne:   # allow the first frame to be missed
                        break
            if len(x)>5: # Only capture lines that matched blobs
                c = {}
                c["x"] = x.copy()
                c["y"] = y.copy()
                c["i"] = idata.copy()
                cometsFound.append(c)

            for a in range(1+matches):
                x.pop()
                y.pop()
                timestamp.pop()
                idata.pop()                    
           
    return cometsFound

def explore_sequence(seq):
    """
        Extract the comets from a given sequence
    """
    if DEBUG:
        print("Sequence: " + seq["ID"])
    # number of images
    numImg = len(seq["path"]) 
    if DEBUG:
        print("Number of images: "+str(numImg))

    width = 1024
    height = 1024

    # Create 3D data cube to hold data, assuming all data have
    # array sizes of 1024x1024 pixels.
    data_cube = np.empty((width,height,numImg))

    timestamps = [0] * numImg

    for i in range(numImg):
        # read image and header from FITS file
        img, hdr = fits.getdata(seq["path"][i], header=True)
        
        # Normalize by exposure time (a good practice for LASCO data)
        img = img.astype('float64') / hdr['EXPTIME']
        
        # Collect timestamps
        timestamps[i] = hdr["MID_TIME"]
        
        # Store array into datacube (3D array)
        data_cube[:,:,i] = img - ndimage.median_filter(img, size=9)# - medfilt2d(img, kernel_size=9)
    
    timestamps.append(0)
    seq["time"] = timestamps
    # Take the difference between consecutive images
    rdiff = np.diff(data_cube, axis=2)
    if DEBUG:
        print("Images loaded")
    comets = []
    for i in range(numImg-1):
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
        blobs_log = blob_log(medsub, max_sigma=5, min_sigma=1, num_sigma=5, threshold=0.1)
        if len(blobs_log)>0:
            blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
        comets.append(blobs_log)

    cometsFiltered = []
    x = []
    y = []
    tm = []
    idata = []
    dist = 0
    numStr = ""
    maxNum = 0
    for c in comets:
        numStr += str(len(c))+" "
        maxNum = max(maxNum, len(c))
    if DEBUG:
        print("Number of possible comets for each frame: " + numStr)
    if maxNum<1000: # Limit to max of 1000 of each frame, if it goes above this the thresholds are probably not set correctly
        cometsFound = []
        for ss in range(len(comets)-5):
            if ss>=0:
                if DEBUG:
                    print("start index = " + str(ss) + " object tracks found = " + str(len(cometsFound)))
                if len(cometsFound)>20: # exit with many tracks, probably too much noise in frames
                    break
                cometsFound += search_comet(seq, comets, ss, x, y, idata, dist, tm)

        if len(cometsFound)>0:
            if DEBUG:
                print("Removing near duplicates from "+str(len(cometsFound))+" objects")
        
        # Remove near duplicate comets. Would be better to compare lines instead of pairs of points
        removeme = [0] * len(cometsFound)
        for c1 in range(len(cometsFound)):
            len1 = len(cometsFound[c1]["x"])
            for c2 in range(c1+1, len(cometsFound)):
                if removeme[c2]==0 and removeme[c1]==0:
                    len2 = len(cometsFound[c2]["x"])
                    m = 0
                    i1 = 0
                    i2 = 0
                    while i1<len1 and i2<len2:
                        while i1<len1 and cometsFound[c1]["i"][i1]<cometsFound[c2]["i"][i2]:
                            i1+=1
                        if i1<len1:
                            while i2<len2 and cometsFound[c2]["i"][i2]<cometsFound[c1]["i"][i1]:
                                i2+=1
                            if i2<len2:
                                if abs(cometsFound[c1]["x"][i1]-cometsFound[c2]["x"][i2])<=5 and abs(cometsFound[c1]["y"][i1]-cometsFound[c2]["y"][i2])<=5:
                                    m += 1
                                    if m>=3:
                                        break
                                i1+=1
                                i2+=1
                    if m>=3:
                        if len(cometsFound[c1]["x"])<=len(cometsFound[c2]["x"]):
                            removeme[c1] = 1
                            break
                        elif len(cometsFound[c2]["x"])<len(cometsFound[c1]["x"]):
                            removeme[c2] = 1
        
        for c1 in range(len(cometsFound)):
            if removeme[c1]==0:
                cometsFiltered.append(cometsFound[c1])
        if DEBUG:
            print(cometsFiltered)

    return cometsFiltered

def process_sequence(s):
    # Find comets in the sequence and return only the longest matched one
    result = []
    slen = len(s["images"])
    if slen>=5:  # Ignore short sequences
        try:
            cometsFound = explore_sequence(s)
            bestP = -1
            bestComet = ""
            for c in cometsFound:
                try:
                    # Confidence is the percentage of frames in which we detected the comet
                    P = float(len(c["x"])) / slen
                    if P>bestP:
                        bestP = P
                        bestComet = s["ID"] + ","
                        # Start position and time
                        sx = c["x"][0]
                        sy = c["y"][0]
                        stime = s["time"][c["i"][0]]
                        # Direction and speed
                        vx = c["x"][-1] - sx
                        vy = c["y"][-1] - sy
                        vtime = s["time"][c["i"][-1]] - stime 
                        # Output a location for each frame, extrapolate the coordinate if non was detected in the frame
                        for frame in range(slen):
                            # Have we detected something?
                            if frame in c["i"]:
                                idx = c["i"].index(frame)
                                # Yes, output the coordinate
                                imgid = s["path"][frame].split("/")[-1]
                                bestComet += imgid + "," + str(c["x"][idx]) + "," + str(c["y"][idx]) + ","
                            else:
                                # Extrapolate the value based on direction, speed and time
                                dtime = s["time"][frame] - stime
                                px = sx + vx*dtime/vtime
                                py = sy + vy*dtime/vtime
                                if px>=0 and px<1024 and py>=0 and py<1024:
                                    # output only when it's within the frame
                                    imgid = s["path"][frame].split("/")[-1]
                                    bestComet += imgid + "," + str(px) + "," + str(py) + ","
                        bestComet += str(P)+"\n"
                except Exception as e:
                    print("Error: "+str(e))
                    pass
            if bestP>0:
                result.append(bestComet)
        except Exception as e:
            print("Error: "+str(e))
            pass
    print(s["progress"])
    return result



#######################################################################################
folder_in = sys.argv[1]
output_file = sys.argv[2]

data_set = []
# Scan folder for all sequences
for (dirpath, dirnames, filenames) in os.walk(folder_in):
    dirnames.sort()

    seq = {}
    cometID = os.path.relpath(dirpath, folder_in)
    seq["ID"] = cometID
    images = []
    paths = []
    for filename in filenames:
        ext = os.path.splitext(filename)[1]
        if ext=='.fts':
            images.append(filename)
            paths.append(os.path.join(dirpath, filename))

    images.sort()
    paths.sort()
    seq["images"] = images
    seq["path"] = paths
    if len(images)>0:
        data_set.append(seq)

for i, s in enumerate(data_set):
    s["progress"] = "Completed "+s["ID"]+" "+str(i+1)+"/"+str(len(data_set))

pool = multiprocessing.Pool()
result_async = [pool.apply_async(process_sequence, args = (s, )) for s in data_set]
results = [r.get() for r in result_async]
with open(output_file, 'w') as f:
    for r in results:
        if len(r)>0:
            f.writelines(r)
            f.flush()


