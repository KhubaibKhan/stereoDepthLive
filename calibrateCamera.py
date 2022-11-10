# Package importation
import numpy as np
import cv2
import glob
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from datetime import datetime
from open3d import *


dt = datetime.now().strftime("%Y%m%d_%H%M")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
infraredL = cv2.VideoWriter(f'{os.getcwd()}/infraredL_{dt}.mp4', fourcc,
                          30, (640, 480))

out_depth = cv2.VideoWriter(f'{os.getcwd()}/depth_{dt}.mp4', fourcc,
                          30, (640, 480))

infraredR = cv2.VideoWriter(f'{os.getcwd()}/infraredR_{dt}.mp4', fourcc,
                          30, (640, 480))

# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []


imagesLeft = glob.glob('.\calibration_images\*L.png')
imagesRight = glob.glob('.\calibration_images\*R.png')
# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0,len(imagesLeft)):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    ChessImaR= cv2.imread(imagesLeft[i],0)    # Right side
    ChessImaL= cv2.imread(imagesRight[i],0)    # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (9,6),None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (9,6),None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print('Cameras Ready to use')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria = criteria_stereo,
                                                          flags = cv2.CALIB_FIX_INTRINSIC)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)


print("Saving parameters!")
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_WRITE)

cv_file.write('leftStereoMap', Left_Stereo_Map)
cv_file.write('rightStereoMap', Right_Stereo_Map)

cv_file.release()