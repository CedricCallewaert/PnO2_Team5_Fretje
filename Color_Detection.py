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
        
    center = None
    
    if len(contours) > 0:
        #get the max contour
        c = max(contours, key = cv2.contourArea)
        
        # transform into rectangle
        circle = cv2.minEnclosingCircle(c)
        
        ((x, y), radius) = circle
        
        s = "x: {}, y: {}, radius: {}".format(np.round(x), np.round(y), np.round(radius)) 
        print(s)
        
        #box
        center = (int(x),int(y))
        radius = int(radius)
        
        #moment
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # draw contour : yellow
        cv2.circle(frame,center,radius,(0,255,0),2)
        
        # draw a dot to the center : pink
        cv2.circle(frame, center, 5, (255, 0, 255), -1)
        
        # write to the screen
        cv2.putText(frame, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)


    


    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()


cv2.destroyAllWindows()