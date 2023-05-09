import cv2 
import numpy as np

cameraNr = 0
aantalFrames = 5




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

	return x1, y1


def start_stream():
	global cap1
	cap1 = cv2.VideoCapture(cameraNr)

exposure = 0
def change_exposure(x):
	global exposure
	exposure = x

xGem = 0
yGem = 0
def get_coordinates():
	return [xGem, yGem]

previousExposure = None
xList = [0 for i in range(aantalFrames)]
yList = [0 for i in range(aantalFrames)]
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
		x, y = draw_circle(sorted_contours, 0, frame)
		xList = [x] + xList[:-1] #Nieuwe waarde vooraan, laatste waarde weg
		yList = [y] + yList[:-1]
	
	# show video
	cv2.imshow("Frame", frame)

	cv2.waitKey(1) #show the image for x mseconds before it automatically closes

	##### Gemiddelde van laatste waarden #####

	xGem = sum(xList)/len(xList)
	yGem = sum(yList)/len(yList)

def close_stream():
	global cap1
	
	cap1.release()
	cv2.destroyAllWindows()

"""
from time import sleep

start_stream()
print("started")

while True:
	get_frame()
	print(get_coordinates())

print("ended")
close_stream()
"""