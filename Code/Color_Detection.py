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
    
# def calculate_projection_matrix(K, dist_coeffs, three_D_points, two_D_points):
    
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

# def get_coordinates(num_points):
    
    
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

    
def red_recoginion(frames):

    # initialise arrays

    som1=np.array([0,0], dtype=np.float64)
    teller1=0

    som2=np.array([0,0], dtype=np.float64)
    teller2=0

    som3=np.array([0,0], dtype=np.float64)
    teller3=0

    # start the video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_EXPOSURE,-4)

    _, frame = cap.read()

    # convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    # define range of red color in HSV
    lower_red_1 = np.array([0,0,220]) 
    upper_red_1 = np.array([179,50,255])
    mask = cv2.inRange(hsv, lower_red_1, upper_red_1)

    # find contours + sort them
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    
    for i in range(frames):

        if len(contours) > 0:
            teller1+=1
            som1 += draw_circle(sorted_contours, 0, frame)
            
        if len(contours) > 1:
            teller2+=1
            som2 += draw_circle(sorted_contours, 1, frame)
            
        if len(contours) > 2:
            teller3+=1
            som3 += draw_circle(sorted_contours, 2, frame)
        
    # show video
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    
    # close the video capture
    cap.release()
    cv2.destroyAllWindows()
    return som1/teller1, som2/teller2, som3/teller3

    
    


def main(frames):

   # Get the camera calibration matrix and distortion coefficients
    if input("Do you want to calibrate the camera? (y/n): ") == "y":
        K, dist = camera_calibration()

        if input ("Do you want to save the calibration for later use? (y/n): ") == "y":
            np.savez("calibration.npz", K=K, dist=dist)
        
    elif input("Do you want to load the calibration from a file? (y/n): ") == "y":
        with np.load("calibration.npz") as X:
            K, dist = [X[i] for i in ("K", "dist")]

    # Get the projection matrix
    if input("Do you want to calculate the projection matrix? (y/n): ") == "y":
        projection_matrix = camera_pose_estimation(K, dist)

        if input ("Do you want to save the projection matrix for later use? (y/n): ") == "y":
            np.savez("projection_matrix.npz", projection_matrix=projection_matrix)

    elif input("Do you want to load the projection matrix from a file? (y/n): ") == "y":
            with np.load("projection_matrix.npz") as X:
                projection_matrix = X["projection_matrix"]
    
    # Find the targets
    if input("Do you want to find the targets? (y/n): ") == "y":
        gemiddelde1, gemiddelde2, gemiddelde3 = red_recoginion(frames)

    # calculate the 3D coordinates of the point
    if input("Do you want to calculate the 3D coordinates of the point? (y/n): ") == "y":

        point_3d_1 = (find_3d_point(projection_matrix, gemiddelde1)).tolist()
        point_3d_2 = (find_3d_point(projection_matrix, gemiddelde2)).tolist()
        point_3d_3 = (find_3d_point(projection_matrix, gemiddelde3)).tolist()


     

    return point_3d_1 + point_3d_2 + point_3d_3


main()
