#%%

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
    ret, depth_frame, rgb_frame, l_infrared, r_infrared = dc.get_frame()
    
    if img_counter %10 == 0:
        cv2.imwrite("./calibration_images/calibration_{}_L.png".format(img_counter), l_infrared)
        cv2.imwrite("./calibration_images/calibration_{}_R.png".format(img_counter), r_infrared)
    cv2.imshow("Infrared Left", l_infrared)
    cv2.imshow("Infrared Right", l_infrared)
    img_counter+= 1
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print('done')
        break

cv2.destroyAllWindows()
dc.release()

