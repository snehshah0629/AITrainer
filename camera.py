import cv2
from poseModule import poseDetector
from trainer import Trainer

class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.stats = [0, 0]
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        trainer = Trainer(frame, self.stats)
        self.stats[0], self.stats[1] = trainer.train()
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()