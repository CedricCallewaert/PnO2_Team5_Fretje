import cv2 
import numpy as np
import glob

def distance_to_camera_alternative(height, y_waarde, x_waarde):
    y_graden= y_waarde * (18/720) + 50,2
    x_graden= x_waarde * (75/1280)

def camera_calibration():
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrate the camera using the object points and image points
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print the camera matrix
    return K, dist, rvecs, tvecs
    


def distance_to_camera(knownWidth, focalLength , perWidth):
    # compute and return the distance from the maker to the camera
    if perWidth > 0 :
        return (knownWidth * focalLength) / perWidth
    else:
        return np.nan

def focal_length(measured_distance, real_width, width_in_real):
    focal_length = (width_in_real * measured_distance) / real_width
    return focal_length

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

    # calculate distance
    
    distance1 = distance_to_camera(85, 1425, radius1)
    
    #create text
    s1 = "x1: {}, y1: {}, radius1: {}, distance1: {}".format(np.round(x1), np.round(y1), np.round(radius1), distance1)
    # write to the screen
    cv2.putText(frame, s1, (25, 50 + number * 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    return np.array([x1, y1, distance1])

    



def main(aantal):

    gemiddelde1 = 0
    gemiddelde2 = 0
    gemiddelde3 = 0

    cap = cv2.VideoCapture(0)

    som1=np.array([0,0,0], dtype=np.float64)
    teller1=0

    som2=np.array([0,0,0], dtype=np.float64)
    teller2=0

    som3=np.array([0,0,0], dtype=np.float64)
    teller3=0


    for i in range(aantal):

        _, frame = cap.read()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        
        
        lower_red_1 = np.array([0,0,220]) 
        upper_red_1 = np.array([179,50,255])

        mask = cv2.inRange(hsv, lower_red_1, upper_red_1)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        
        
        if len(contours) > 0:
            teller1+=1
            som1 += draw_circle(sorted_contours, 0, frame)
          
        if len(contours) > 1:
            teller2+=1
            som2 += draw_circle(sorted_contours, 1, frame)
           
        if len(contours) > 2:
            teller3+=1
            som3 += draw_circle(sorted_contours, 2, frame)
           

        

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1)
        
        i += 1



    cap.release()


    cv2.destroyAllWindows()

    if teller1 != 0 or teller2 != 0 or teller3 != 0:
        gemiddelde1 = (som1/teller1).tolist()
        gemiddelde2 = (som2/teller2).tolist()
        gemiddelde3 = (som3/teller3).tolist()

    output= gemiddelde1 + gemiddelde2 + gemiddelde3

    return output


print(camera_calibration())