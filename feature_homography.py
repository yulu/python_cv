#!/usr/bin/env python

'''
Usage
----
feature_homography.py [<video source>]

Keys:
	SPACE - pause video

Select a texture planar object to track by drawing a box with mouse
'''

import numpy as np
import cv2
import video
import common
from common import getsize, draw_keypoints
from plane_tracker import PlaneTracker

def is_rect_nonzero(r):
	(_,_,w,h) = r
	return (w > 0) and (h > 0)

class App:
	def __init__(self, src):
		self.cap = video.create_capture(src)
		self.frame = None
		self.paused = False
		self.tracker = PlaneTracker()

		cv2.namedWindow("plane")
		cv2.setMouseCallback("plane", self.on_mouse)
		self.drag_start = None
		self.track_window = None
		#self.rect_sel = common.RectSelector('plane', self.on_rect)

	def on_mouse(self, event, x, y, flags, param):
		x, y = np.int16([x, y])
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drag_start = (x, y)
		if event == cv2.EVENT_LBUTTONUP:
			self.drag_start = None
			self.track_window = self.selection

			#Set the tracking rect
			self.tracker.clear()
			rect = self.track_window
			self.tracker.add_target(self.frame, rect)
		if self.drag_start:
			xmin = min(x, self.drag_start[0])
			ymin = min(y, self.drag_start[1])
			xmax = max(x, self.drag_start[0])
			ymax = max(y, self.drag_start[1])
			self.selection = (xmin, ymin, xmax, ymax)

	def run(self):
		while True:
			playing = not self.paused and not self.drag_start
			if playing or self.frame is None:
				ret, frame = self.cap.read()
				if not ret:
					break
				self.frame = frame.copy()

			w, h = getsize(self.frame)
			vis = np.zeros((h, w*2, 3), np.uint8)
			vis[:h, :w] = self.frame
			if len(self.tracker.targets) > 0:
				target = self.tracker.targets[0]
				vis[:, w:] = target.image
				draw_keypoints(vis[:, w:], target.keypoints)
				x0, y0, x1, y1 = target.rect
				cv2.rectangle(vis, (x0+w, y0), (x1+w, y1), (0, 255, 0), 2)

			if playing:
				tracked = self.tracker.track(self.frame)
				if len(tracked) > 0:
					tracked = tracked[0]
					cv2.polylines(vis, [np.int32(tracked.quad)], True, (255,255,255), 2)
					for (x0, y0), (x1, y1) in zip(np.int32(tracked.p0), np.int32(tracked.p1)):
						cv2.line(vis, (x0+w, y0), (x1, y1), (0, 255, 0))

			draw_keypoints(vis, self.tracker.frame_points)

			#self.rect_sel.draw(vis)
			if self.drag_start:
				x, y, x1, y1 = self.selection
				cv2.rectangle(vis, (x, y), (x1, y1), (0, 255, 0), 2)

			cv2.imshow('plane', vis)
			ch = cv2.waitKey(1)
			if ch == ord(' '):
				self.paused = not self.paused
			if ch == 27:
				break

if __name__=='__main__':
	print __doc__

	import sys
	try: video_src = sys.argv[1]
	except: video_src=0
	App(video_src).run()