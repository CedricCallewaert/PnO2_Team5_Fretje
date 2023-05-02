import cv2 
import numpy as np
import glob
from time import sleep
import os

def draw_circle(contours, number, frame):
    
    c = contours[int(number)]
    # transform into circle
    circle1 = cv2.minEnclosingCircle(c)
    ((x1, y1), radius1) = circle1

    #circle
    center1 = (int(x1),int(y1))
    radius1 = int(radius1)

    #moment
    M1 = cv2.moments(c)
    if M1["m00"] == 0:
        M1["m00"] = 1.0
    center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))

    # draw contour : yellow
    cv2.circle(frame,center1,radius1,(0,255,0),2)

    # draw a dot to the center : pink
    cv2.circle(frame, center1, 5, (255, 0, 255), -1)

    #create text
    s1 = "x1: {}, y1: {}, radius1: {}".format(np.round(x1), np.round(y1), np.round(radius1))
    # write to the screen
    cv2.putText(frame, s1, (25, 50 + number * 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    return np.array([x1, y1])

def draw_circle_background(contours, number, frame):
    
    c = contours[int(number)]
    # transform into circle
    circle1 = cv2.minEnclosingCircle(c)
    ((x1, y1), radius1) = circle1

    #moment
    M1 = cv2.moments(c)
    if M1["m00"] == 0:
        M1["m00"] = 1.0
    center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))

    return np.array(center1)

def start_stream():
    global cap1
    cap1 = cv2.VideoCapture(0)
    cap1.set(cv2.CAP_PROP_EXPOSURE,-4)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)

def get_frame():
    
    _, frame = cap1.read()
    # convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    # define range of red color in HSV
    lower_red_1 = np.array([0,0,220]) 
    upper_red_1 = np.array([179,50,255])
    mask = cv2.inRange(hsv, lower_red_1, upper_red_1)

    # find contours + sort them
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    
    
    if len(contours) > 0:
        draw_circle(sorted_contours, 0, frame)
    
    # show video
    cv2.imshow("Frame", frame)

def close_stream():
    cap1.release()
    cv2.destroyAllWindows()
    
def red_recoginion():

    # initialise array

    som1=np.array([0,0], dtype=np.float64)
    teller1=0

    start_time = cv2.getTickCount()

    # Loop to capture frames for the specified duration
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 2:
    # read frame
        _, frame = cap1.read()

        # convert to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        
        # define range of red color in HSV
        lower_red_1 = np.array([0,0,220]) 
        upper_red_1 = np.array([179,50,255])
        mask = cv2.inRange(hsv, lower_red_1, upper_red_1)

        # find contours + sort them
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        

        if len(contours) > 0:
            teller1+=1
            som1 += draw_circle_background(sorted_contours, 0, frame)

    
    # close the video capture
    return som1/teller1

def calculate_homogeneous_matrix():
    points_real = np.array([[80,80,0],
                            [80,680,0],
                            [680,680,0],
                            [680,80,0]])
        
    points_camera = np.zeros((4, 2)) 

    click_test()

    for i in range(4):
        filename = f"points/calibration_point{i}.npz"
        data = np.load(filename)
        point = data["calibration_point"]
        points_camera[i] = point
   
    homography_matrix,_ = cv2.findHomography(points_camera,points_real )
    np.savez("homography.npz", homography_matrix=homography_matrix)
    
def click_event(event,x,y,flags,params):
    global ii
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_point= np.array([x,y])
        print(calibration_point)
        np.savez(f"points/calibration_point{ii}.npz", calibration_point=calibration_point)
        ii+=1

def click_test():
    global ii
    ii=0
    cv2.namedWindow("Video Capture")
    cv2.setMouseCallback("Video Capture", click_event)
    while True:
        ret, frame = cap1.read()
        cv2.imshow("Video Capture", frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

def calculate_angles(point_3D):
    # calculate angles
    x = abs(point_3D[0]-3)
    y = point_3D[1]+2.5

    # calculate angles
    alpha = np.arctan2(y, x)
 
    # calculate distance
    distance_plane = np.sqrt(x**2 + y**2)
    distance_direct = np.sqrt(distance_plane**2 + (2.5)**2)

    # convert to degrees
    alpha = np.rad2deg(alpha)
 

    return [alpha, distance_direct]

def main():
    with np.load("homography.npz") as X:
        H = X["homography_matrix"]
   
    point_2D = red_recoginion()

    point_2D=  np.append(point_2D,1)
    point_3D = np.dot(H, point_2D)

    output = calculate_angles(point_3D)

    return output
