# Imports
import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector():

    # Constructor
    def __init__(self, mode = False, smooth = True, detection_confidence = 0.5, tracking_confidence = 0.5):
        self.mode = mode
        self.smooth = smooth
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.smooth, self.detection_confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    # Find Pose function
    def find_pose(self, img, draw = True):
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Run through model
        self.results = self.pose.process(img_rgb)

        # If draw = True and landmarks exist, draw them and overlay on the points
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    # Draw Positions function
    def get_position(self, img, draw = True):
        # List of (x, y, id) values for each landmark
        self.lm_list = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                self.lm_list.append([id, cx, cy])
                if draw:        
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lm_list

    def get_slope(self, img, points_list, draw = True):
        p1, p2 = points_list
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        slope = np.abs((y2 - y1) / (x2 - x1 + 1e-10))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
        cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
        cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str(int(slope)), (x2 - 75, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 3)
        return slope


    def get_angle(self, img, points_list, draw = True):
        p1, p2, p3 = points_list
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]
        m1 = (y1 - y2) / (x1 - x2 + 1e-10)
        m2 = (y3 - y2) / (x3 - x2 + 1e-10)
        ang_r = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        ang_d = round(math.degrees(ang_r))
        if ang_d < 0:
            ang_d = -1 * ang_d
        else:
            ang_d = 360 - ang_d
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, str(ang_d), (x2 - 75, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        return ang_d

def main():

    bicep1 = 'bicepCurl1.mp4'
    raas = 'raas.mp4'

    path = 'videos/' + bicep1
    # Get video
    cap = cv2.VideoCapture(path)
    pastTime = 0
    detector = poseDetector()

    while True:
        # Read video
        success, img = cap.read()

        # Calculate and show fps
        currTime = time.time()
        fps = 1 / (currTime - pastTime)
        pastTime = currTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Track pose
        img = detector.find_pose(img)

        # Get landmark positions
        lm_list = detector.get_position(img, draw = False)

        # To track specific landmarks
        if len(lm_list) > 0:
            left_arm = [11, 13, 15]
            detector.get_angle(img, left_arm)

        # Show image
        cv2.imshow('Image', img)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()