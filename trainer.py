import cv2
import numpy as np
import time
import math
from poseModule import poseDetector

class Trainer():

    def __init__(self, img, stats):
        self.detector = poseDetector()
        self.img = img
        self.detector.find_pose(img, draw = False)
        self.lm_list = self.detector.get_position(img, False)
        self.count = stats[0]
        self.dir = stats[1]

    def train(self):
        if len(self.lm_list) > 0:
            self.count, self.dir = self.bicep_curl()
        return self.count, self.dir

    def bicep_curl(self):
        left_arm = [11, 13, 15]
        right_arm = [12, 14, 16]
        left_back = [11, 23]
        right_back = [12, 24]

        count = self.count
        dir = self.dir

        angle = self.detector.get_angle(self.img, left_arm)
        back_slope = self.detector.get_slope(self.img, left_back)
        percentage = np.interp(angle, (50, 150), (100, 0))
        bar = 500 - np.interp(angle, (50, 150), (400, 100))

        # Check for dumbbell curls
        if percentage == 100:
            if dir == 1 and back_slope < 15:
                count += 1
                dir = 0
        if percentage == 0:
            if dir == 0 and back_slope < 15:
                dir = 1
        if back_slope >= 15:
            text = "Straighten your back!!"
            cv2.putText(self.img, text, (int(self.img.shape[1] / 4), int(self.img.shape[0] - 250)), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
    
        # Draw Bar
        cv2.rectangle(self.img, (250, 100), (350, 400), (0, 255, 0), 3)
        cv2.rectangle(self.img, (250, int(bar)), (350, 400), (0, 255, 0), cv2.FILLED)
        #Curl Count
        cv2.putText(self.img, str(int(self.count)), (75, 150), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 0, 0), 5)
        return count, dir

def main():

    cap = cv2.VideoCapture("videos/curls.mp4")
    img = cv2.imread("videos/test.jpg")
    detector = pm.poseDetector()

    count = 0
    dir = 1

    while True:
        success, img = cap.read()
        detector.find_pose(img, draw = False)
        lm_list = detector.get_position(img, False)

        if len(lm_list) > 0:
            count, dir = bicep_curl(img, detector, count, dir)

    
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()