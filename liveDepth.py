# Package importation
import numpy as np
import cv2
import glob
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from datetime import datetime
from open3d import *

cv2.ximgproc
# Filtering
kernel= np.ones((3,3),np.uint8)
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    '''

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        zDepth = (50*8)/average
        print('Distance: '+ str(zDepth/3289)+' m')
        
# This section has to be uncommented if you want to take mesurements and store them in the excel
##        ws.append([counterdist, average])
##        print('Measure at '+str(counterdist)+' cm, the dispasrity is ' + str(average))
##        if (counterdist <= 85):
##            counterdist += 3
##        elif(counterdist <= 120):
##            counterdist += 5
##        else:
##            counterdist += 10
##        print('Next distance to measure: '+str(counterdist)+'cm')
dt = datetime.now().strftime("%Y%m%d_%H%M")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
infraredL = cv2.VideoWriter(f'{os.getcwd()}/videos/infraredL_{dt}.mp4', fourcc,
                          30, (640, 480))

out_depth = cv2.VideoWriter(f'{os.getcwd()}/videos/depth_{dt}.mp4', fourcc,
                          30, (640, 480))

infraredR = cv2.VideoWriter(f'{os.getcwd()}/videos/infraredR_{dt}.mp4', fourcc,
                          30, (640, 480))

# Mouseclick callback
wb=Workbook()
ws=wb.active  
#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

Q = cv_file.getNode("Q").mat()
Left_Stereo_Map0 = cv_file.getNode('leftStereoMap0').mat()
Left_Stereo_Map1 = cv_file.getNode('leftStereoMap1').mat()
Right_Stereo_Map0 = cv_file.getNode('rightStereoMap0').mat()
Right_Stereo_Map1 = cv_file.getNode('rightStereoMap1').mat()

Left_Stereo_Map = (Left_Stereo_Map0, Left_Stereo_Map1)
Right_Stereo_Map = (Right_Stereo_Map0, Right_Stereo_Map1)
# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


#*************************************
#***** Starting the StereoVision *****
#*************************************
import os
from datetime import datetime

import cv2
import pyrealsense2
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from realsense_depth import DepthCamera
# Get the color map by name:
cm = plt.get_cmap('jet')
dt = datetime.now().strftime("%Y%m%d_%H%M")
dc = DepthCamera()
img_counter = 0
while True:

    ret, depth_frame, rgb_frame, frameL, frameR = dc.get_frame()
    depth_abs = cv2.convertScaleAbs(depth_frame, alpha=0.03)
    depth_cmap = cv2.applyColorMap(depth_abs, cv2.COLORMAP_JET)

    # Rectify the images on rotation and alignement
    Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the kalibration parameters founds during the initialisation
    Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

    cv2.imshow('Left image',frameL)
    cv2.imshow('Right image',frameR)
        
    grayR = Right_nice
    grayL = Left_nice

    # Compute the 2 images for the Depth_image
    disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
    dispL= disp
    dispR= stereoR.compute(grayR,grayL)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

    # Using the WLS filter
    filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)
    disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

    # Filtering the Results with a closing filter
    closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

    # Colors map
    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

    cv2.imshow('Filtered Color Depth',filt_Color)


    infraredL.write(cv2.cvtColor(grayL,cv2.COLOR_GRAY2RGB))
    infraredR.write(cv2.cvtColor(grayR,cv2.COLOR_GRAY2RGB))
    out_depth.write(filt_Color)

    # Mouse click
    cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)
    
    img_counter+= 1
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print('done')
        break

cv2.destroyAllWindows()
dc.release()

