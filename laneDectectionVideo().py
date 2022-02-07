from linecache import getlines
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def region_of_interest(image):
    height = image.shape[0] 
    width = image.shape[1]
    polygon = np.array([[(int(width/6), height), (int(width/2.5), int(height/1.45)),(int(width/1.9), int(height/1.45)),(int(width/1.3), height)]])
    
    # create 0 array size of image
    zeroMask = np.zeros_like(image)
    
    # fill polygon with 1
    cv2.fillConvexPoly(zeroMask,polygon,1)
   
    # mask our original image to create our own region of interest
    roi = cv2.bitwise_and(image,image,mask = zeroMask)
    return roi

def get_line_coordinates(frame,lines):
    height = int(frame.shape[0]/1.5)
    slope,y_intercept = lines[0], lines[1]

    # get first bottom y-coordinate
    y1 = frame.shape[0]

    # get second top y-coordinate
    y2 = int(y1 - 110)

    # first x-coordinate, solving for x in y = mx + b
    x1 = int((y1-y_intercept)/slope)

    # second x-coordinate
    x2 = int((y2-y_intercept)/slope)
    return np.array([x1,y1,x2,y2])

# average outline from hough transformation

def get_lines(frame,lines):
    copy_image = frame.copy()
    left_line , right_line = [], []
    roi_height = int(frame.shape[0]/1.5)
    line_frame = np.zeros_like(frame)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # calculate slope and y intercept
        line_data = np.polyfit((x1,x2), (y1, y2), 1)
        slope , y_intercept = round(line_data[0], 8),line_data[1]
        if slope < 0:
            left_line.append((slope,y_intercept))
        else: 
            right_line.append((slope, y_intercept))

    if left_line :
        left_line_average = np.average(left_line, axis = 0)
        left = get_line_coordinates(frame, left_line_average)
        try: 
            cv2.line(line_frame,(left[0], left[1]), (left[2], left[3]), (255,0,0), 2)
        except Exception as e:
            print('Error', e)
    
    if right_line:
        right_line_average = np.average(right_line, axis = 0)
        right = get_line_coordinates(frame, right_line_average)
        try: 
            cv2.line(line_frame,(right[0], right[1]), (right[2], right[3]), (255,0,0), 2)
        except Exception as e:
            print('Error', e)
    return cv2.addWeighted(copy_image, 0.8, line_frame, 0.8, 0.0)

# save finalized video 
four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('media/output.mp4', four_cc, 20.0, (640, 360))
vid = cv2.VideoCapture('media/carDashCam.mp4')
count = 0

while(vid.isOpened()):
    ret,frame = vid.read()
    cv2.imshow('Original frame', frame)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kernel_size = 1
    gaussian_frame = cv2.GaussianBlur(gray_frame,(kernel_size, kernel_size), kernel_size)
    cv2.imshow('Gaussian Frame', gaussian_frame)
    low_threshold = 75
    high_threshold = 160
    edge_frame = cv2.Canny(gaussian_frame, low_threshold, high_threshold)
    roi_frame = region_of_interest(edge_frame)
    lines = cv2.HoughLinesP(roi_frame, rho = 1, theta = np.pi/180, threshold = 20, lines = np.array([]), minLineLength = 10, maxLineGap = 180)
    image_with_lines = get_lines(frame, lines)
    cv2.imshow('Final', image_with_lines)
    out.write(image_with_lines)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()