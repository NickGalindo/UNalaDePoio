from .preprocess.preprocess import splitVideo

def detect_objects_in_video(video_path, output_path):
    splitVideo(video_path, output_path)
