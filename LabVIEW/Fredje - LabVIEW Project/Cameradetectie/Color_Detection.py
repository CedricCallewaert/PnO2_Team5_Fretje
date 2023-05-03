import cv2 
import numpy as np
import glob
from time import sleep
import os

# global variables
cameraNr = 0
exposure = 0
aantalFrames = 5
previousExposure = None
xList = [0 for i in range(aantalFrames)]
yList = [0 for i in range(aantalFrames)]

def get_coordinates():
    # load the homography matrix
    with np.load("homography.npz") as X:
        H = X["homography_matrix"]

    # calculate average of x and y coordinates
    xGem = sum(xList)/len(xList)
    yGem = sum(yList)/len(yList)

    # calculate 3D coordinates
    point_2D=np.array([xGem, yGem, 1])
    point_3D = np.dot(H, point_2D)

    output = calculate_angles(point_3D)

    return output, [xGem, yGem]

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

    return center1

def start_stream():
    global cap1
    cap1 = cv2.VideoCapture(cameraNr)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)

def change_exposure(x):
	global exposure
	exposure = x
        


def get_frame():
    global cap1
    global previousExposure
    global xList, yList, xGem, yGem

    if exposure != previousExposure:
        print(f"Exposure setting now {exposure}")
        previousExposure = exposure
        _, frame = cap1.read()
        cap1.set(cv2.CAP_PROP_EXPOSURE,exposure)

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
        xy = draw_circle(sorted_contours, 0, frame)
        xList = [xy[0]] + xList[:-1]
        yList = [xy[1]] + yList[:-1]

    # show video
    cv2.imshow("Frame", frame)

    #show the image for x mseconds before it automatically closes
    cv2.waitKey(1) 

def close_stream():
    global cap1
    cap1.release()
    cv2.destroyAllWindows()

def calculate_homogeneous_matrix():
    points_real = np.array([[2320/7,0,0],
                            [2320/7,720,0],
                            [6640/7,720,0],
                            [6640/7,0,0]])
        
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
    # transform from pixels to meters
    x=(point_3D[0]*6)/(4320/7)-3
    y=(point_3D[1]*7)/720

    # calculate angles
    alpha = np.arctan2(y, x)
 
    # calculate distance
    distance_plane = np.sqrt(x**2 + y**2)
    distance_direct = np.sqrt(distance_plane**2 + (2.9)**2)

    # convert to degrees
    alpha = np.rad2deg(alpha)
 

    return [alpha, distance_plane]

