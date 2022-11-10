# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:42:29 2022

@author: talha
"""
import cv2                            
import numpy as np                        
import matplotlib.pyplot as plt          
import pyrealsense2 as rs                 
import pandas as pd   



bag_file = 'E:/20221021_150134.bag'

depth, rgb = [], []

config = rs.config()
rs.config.enable_device_from_file(config, bag_file)
pipeline = rs.pipeline()
profile = pipeline.start(config)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,1)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 1)

align_to = rs.stream.color
align = rs.align(align_to)

stream = True

i = 0
while stream:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        if i%60 == 0:
            depth.append(depth_image)
            rgb.append(color_image)
    
        i += 1
        print(f'Frames read: {i}; Appended frames: {i//30}')
        if not depth_frame or not color_frame:
            stream = False
            pipeline.stop()
        
        
depth_abs = cv2.convertScaleAbs(depth_image, alpha=0.03)
depth_cmap = cv2.applyColorMap(depth_abs, cv2.COLORMAP_JET)          

plt.figure()
plt.imshow(color_image) 
plt.figure()
plt.imshow(depth_cmap) 
            
