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
from collections import namedtuple
from common import getsize, draw_keypoints

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

'''
  image     - image to track
  rect      - tracked rectangle (x1, y1, x2, y2)
  keypoints - keypoints detected inside rect
  descrs    - their descriptors
  data      - some user-provided data
'''
PlanarTarget = namedtuple('PlanarTarget', 'image, rect, keypoints, descrs, data')

'''
  target - reference to PlanarTarget
  p0     - matched points coords in target image
  p1     - matched points coords in input frame
  H      - homography matrix from p0 to p1
  quad   - target bounary quad in input frame
'''
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')

def is_rect_nonzero(r):
	(_,_,w,h) = r
	return (w > 0) and (h > 0)

class PlaneTracker:
    def __init__(self):
        self.detector = cv2.ORB( nfeatures = 1000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        self.targets = []

    def add_target(self, image, rect, data=None):
        '''Add a new tracking target.'''
        x0, y0, x1, y1 = rect
        raw_points, raw_descrs = self.detect_features(image)
        points, descs = [], []
        for kp, desc in zip(raw_points, raw_descrs):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)
        if len(points) < MIN_MATCH_COUNT:
        	return []
        descs = np.uint8(descs)
        self.matcher.add([descs])
        target = PlanarTarget(image = image, rect=rect, keypoints = points, descrs=descs, data=None)
        self.targets.append(target)

    def clear(self):
        '''Remove all targets'''
        self.targets = []
        self.matcher.clear()

    def track(self, frame):
        '''Returns a list of detected TrackedTarget objects'''
        self.frame_points, self.frame_descrs = self.detect_features(frame)
        if len(self.frame_points) < MIN_MATCH_COUNT:
            return []
        matches = self.matcher.knnMatch(self.frame_descrs, k = 2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []
        matches_by_id = [[] for _ in xrange(len(self.targets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)

        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < MIN_MATCH_COUNT:
                continue
            target = self.targets[imgIdx]
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [self.frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < MIN_MATCH_COUNT:
                continue
            p0, p1 = p0[status], p1[status]

            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked

    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs

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