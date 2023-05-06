import cv2 
import numpy as np
import glob
from time import sleep
import os


# global variables
cameraNr = 0
exposure = 0
previousExposure = None


def get_coordinates():
    # load the homography matrix
    with np.load("homography.npz") as X:
        H = X["homography_matrix"]

    # calculate 3D coordinates
    point_2D=np.array([xGem, yGem, 1])
    point_3D = np.dot(H, point_2D)
    point_3D /= point_3D[2]

    output = calculate_angles(point_3D)
    

    return output, [xGem, yGem], point_3D

def draw_circle(contour, frame):
    
    c = contour
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

    return center1[0], center1[1]

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
    global x1, x2, x3, x4, x5, x6, x7, x8, x9
    global y1, y2, y3, y4, y5, y6, y7, y8, y9
    global contours
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
        x1, y1 = draw_circle(sorted_contours[0], frame)
    if len(contours) > 1:
        x2, y2 = draw_circle(sorted_contours[1], frame)
    if len(contours) > 2:
        x3, y3 = draw_circle(sorted_contours[2], frame)
    if len(contours) > 3:
        x4, y4 = draw_circle(sorted_contours[3], frame)
    if len(contours) > 4:
        x5, y5 = draw_circle(sorted_contours[4], frame)
    if len(contours) > 5:
        x6, y6 = draw_circle(sorted_contours[5], frame)
    if len(contours) > 6:
        x7, y7 = draw_circle(sorted_contours[6], frame)
    if len(contours) > 7:
        x8, y8 = draw_circle(sorted_contours[7], frame)
    if len(contours) > 8:
        x9, y9 = draw_circle(sorted_contours[8], frame)

    


    #show the image for x mseconds before it automatically closes
    #cv2.waitKey(1) 
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_triplets():
    coordinates= [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8), (x9, y9)]
    coordinates= coordinates[:len(contours)]
    close_triplets = []
    threshold = 80

    # Loop over every possible combination of three points
    for i in range(len(coordinates)):
        for j in range(i+1, len(coordinates)):
            for k in range(j+1, len(coordinates)):
                # Calculate the distances between the three points
                d1 = distance(coordinates[i], coordinates[j])
                d2 = distance(coordinates[i], coordinates[k])
                d3 = distance(coordinates[j], coordinates[k])
                # If all three distances are less than the threshold, add this triplet to the list
                if d1 < threshold and d2 < threshold and d3 < threshold:
                    close_triplets.append((coordinates[i], coordinates[j], coordinates[k]))



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
def warp():
    # Capture a frame
    

    with np.load("homography.npz") as X:
        H = X["homography_matrix"]
    
    while True:
        ret, frame = cap1.read()
        frame_warp = cv2.warpPerspective(frame, H,(frame.shape[1],frame.shape[0]))
        cv2.imshow("warp", frame_warp)
        key = cv2.waitKey(1)
        if key == 27:
            break
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
    x=(point_3D[0]*7)/720-56/9
    y=7-(point_3D[1]*7)/720+3

    # calculate angles
    alpha = np.arctan2(x, y)
 
    # calculate distance
    distance_plane = np.sqrt(x**2 + y**2)
    distance_direct = np.sqrt(distance_plane**2 + (2.9)**2)

    # convert to degrees
    alpha = np.rad2deg(alpha)
 

    return [alpha, distance_plane], [x,y]


calculate_homogeneous_matrix()


            