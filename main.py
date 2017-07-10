import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from global_vars import GV
from model_trainer import ModelTrainer
from car_detector import CarDetector

def process_image(frame):

    image=np.empty_like(frame)
    np.copyto(image,frame)
    model = ModelTrainer.get_trained_model()
    img = CarDetector.process(model,image)

    GV.current_frame+=1

    return img


if __name__ == '__main__':

    video = VideoFileClip("project_video.mp4").subclip("00:00:41","00:00:42")
    annotated_video = video.fl_image(process_image)
    annotated_video.write_videofile("out.mp4", audio=False)

