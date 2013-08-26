#!/usr/bin/python

import cv2.cv as cv
import time

cv.NamedWindow("camera", 1)

capture = cv.CaptureFromCAM(0)

while True:
	#get camera frame
	frame = cv.QueryFrame(capture)

	#convert color frame to gray
	gray = cv.CreateImage(cv.GetSize(frame), 8, 1)
	cv.CvtColor(frame, gray, cv.CV_BGR2GRAY)

	#display the gray frame
	cv.ShowImage("camera", gray)

	#break when esc key is pressed
	if cv.WaitKey(10) == 27:
		break

cv.DestroyAllWindows()