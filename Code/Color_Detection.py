import cv2 
import numpy as np
import glob


def getImages():
    cap = cv2.VideoCapture(0)

    num = 0

    while cap.isOpened():

        _, img = cap.read()

        k = cv2.waitKey(5)

        if k == 27:
            break
        elif k == ord('s'): # s = save image to file  
            cv2.imwrite('images/img' + str(num) + '.png', img)
            print("image saved!")
            num += 1

        cv2.imshow('Img',img)

    # Release and destroy all windows before termination
    cap.release()

    cv2.destroyAllWindows()

def camera_pose_estimation(K,dist):

    cap = cv2.VideoCapture(0)
    _, img = cap.read()

    # Release and destroy all windows before termination
    cap.release()

    # chessboard parameters
    chessboardSize = (8, 6)
    squareSize = 21
    frameSize = (1920, 1080)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((chessboardSize[0] * chessboardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    objp = objp * squareSize

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Find the chess board corners
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize ,None)

    success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    
    rot_mat, _ = cv2.Rodrigues(rvec)
    
    projection_matrix = np.hstack((rot_mat, tvec))
    
    return projection_matrix
    

def camera_calibration():
    # chessboard parameters
    chessboardSize = (8, 6)
    squareSize = 21
    frameSize = (1920, 1080)



    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((chessboardSize[0] * chessboardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    objp = objp * squareSize

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('images/*.png')
    
    num = 0

    for frame in images:
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize ,None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboardSize , corners2,ret)
            cv2.imwrite('images/img_calculated' + str(num) + '.png', img)
            cv2.imshow('img',img)
            cv2.waitKey(100)
            num += 1
    
    
    cv2.destroyAllWindows()

    
    # Calibrate the camera using the object points and image points
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    
   
    return K, dist
    
def calculate_projection_matrix(K, dist_coeffs, three_D_points, two_D_points):
    
    # Calculate the projection matrix using solvePnP
    
    success, rvec, tvec = cv2.solvePnP(three_D_points, two_D_points, K, dist_coeffs)
    
    rot_mat, _ = cv2.Rodrigues(rvec)
    
    projection_matrix = np.hstack((rot_mat, tvec))
    
    return projection_matrix


def find_3d_point(projection_matrix, image_point):

    # Calculate the 3D coordinates of the point

    point_3d_hom = np.dot(np.linalg.inv(projection_matrix), np.vstack((image_point[0], np.ones(1))))
    
    point_3d = point_3d_hom[:3] / point_3d_hom[3]
    
    return point_3d

def get_coordinates(num_points):
    
    
    coords_3d = np.empty((num_points, 3))
    coords_2d = np.empty((num_points, 2))

    
    for i in range(num_points):
        
        x = float(input(f"Enter X coordinate of point {i+1}: "))
        y = float(input(f"Enter Y coordinate of point {i+1}: "))
        z = float(input(f"Enter Z coordinate of point {i+1}: "))
        coords_3d[i] = np.array([x, y, z])

        
        u = float(input(f"Enter u coordinate of point {i+1}: "))
        v = float(input(f"Enter v coordinate of point {i+1}: "))
        coords_2d[i] = np.array([u, v])

    return coords_3d, coords_2d

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

    #create text
    s1 = "x1: {}, y1: {}, radius1: {}".format(np.round(x1), np.round(y1), np.round(radius1))
    # write to the screen
    cv2.putText(frame, s1, (25, 50 + number * 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    return np.array([x1, y1])

    



def main():
    # Get the camera calibration matrix and distortion coefficients
    K= np.array(([[1.48635604e+03, 0.00000000e+00, 9.79901544e+02],
       [0.00000000e+00, 1.48230147e+03, 4.95819480e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))

    dist= np.array([[ 0.06309409, -0.20227065, -0.00725773,  0.00190645,  0.33597095]])


    # Calculate the projection matrix
    projection_matrix = np.array([[ 1.04390024e-01, -9.94436049e-01 , 1.41303371e-02 , 7.94931404e+01],[ 9.77768968e-01 , 1.05217649e-01 , 1.81375553e-01, -1.38314246e+02],[-1.81853149e-01 ,-5.11759323e-03 , 9.83312383e-01,  1.15914001e+03]])


    gemiddelde1 = 0
    gemiddelde2 = 0
    gemiddelde3 = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_EXPOSURE,-4)

    som1=np.array([0,0,0], dtype=np.float64)
    teller1=0

    som2=np.array([0,0,0], dtype=np.float64)
    teller2=0

    som3=np.array([0,0,0], dtype=np.float64)
    teller3=0


    

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

    
    
    for i in range(3):
        # calculate the 3D coordinates of the point
        point_3d_1 = (find_3d_point(projection_matrix, gemiddelde1)).tolist()
        point_3d_2 = (find_3d_point(projection_matrix, gemiddelde2)).tolist()
        point_3d_3 = (find_3d_point(projection_matrix, gemiddelde3)).tolist()


        

    output= point_3d_1 + point_3d_2 + point_3d_3

    return output


print(main())
