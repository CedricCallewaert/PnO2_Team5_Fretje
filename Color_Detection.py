import cv2 
import numpy as np



cap = cv2.VideoCapture(1)
while True:

    _, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    
    lower_red = np.array([-10,200,0]) 
    upper_red = np.array([2,500,500])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
   
    res = cv2.bitwise_and(frame,frame, mask= mask)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)

    center = None
    
    if len(contours) > 0:

        ###first circle
        #get the max contour
        c = sorted_contours[0]

        # transform into circle
        circle1 = cv2.minEnclosingCircle(c)
        ((x1, y1), radius1) = circle1
        s1 = "x1: {}, y1: {}, radius1: {}".format(np.round(x1), np.round(y1), np.round(radius1))

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

        # write to the screen
        cv2.putText(frame, s1, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)




    if len(contours) > 1:



        ###second circle
        #get the max contour
        d = sorted_contours[1]

        # transform into circle
        circle2 = cv2.minEnclosingCircle(d)
        ((x2, y2), radius2) = circle2
        s2 = "x2: {}, y2: {}, radius2: {}".format(np.round(x2), np.round(y2), np.round(radius2)) 

        #circle
        center2 = (int(x2),int(y2))
        radius2 = int(radius2)

        #moment
        M2 = cv2.moments(d)
        if M2["m00"] == 0:
            M2["m00"] = 1.0
        center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))

        # draw contour : yellow
        cv2.circle(frame,center2,radius2,(0,255,0),2)

        # draw a dot to the center : pink
        cv2.circle(frame, center2, 5, (255, 0, 255), -1)

        # write to the screen
        cv2.putText(frame, s2, (25, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)



    if len(contours) > 2:


        ###third circle
        #get the max contour
        e = sorted_contours[2]

        # transform into circle
        circle3 = cv2.minEnclosingCircle(e)
        ((x3, y3), radius3) = circle3
        s3 = "x3: {}, y3: {}, radius3: {}".format(np.round(x3), np.round(y3), np.round(radius3)) 

        #circle
        center3 = (int(x3),int(y3))
        radius3 = int(radius3)

        #moment
        M3 = cv2.moments(e)
        if M3["m00"] == 0:
            M3["m00"] = 1.0
        center3 = (int(M3["m10"] / M3["m00"]), int(M3["m01"] / M3["m00"]))

        # draw contour : yellow
        cv2.circle(frame,center3,radius3,(0,255,0),2)

        # draw a dot to the center : pink
        cv2.circle(frame, center3, 5, (255, 0, 255), -1)

        # write to the screen
        cv2.putText(frame, s3, (25, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()


cv2.destroyAllWindows()