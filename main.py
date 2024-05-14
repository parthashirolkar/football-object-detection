from utils.video_utils import read_video, save_video
from trackers.tracking import Tracker




def main():
    video_frames = read_video("input_videos/019d5b34_0.mp4")
    tracker = Tracker('models/best.pt')
    tracker.get_object_tracks(video_frames)


    
    # save_video(video_frames, "output_videos/output_video.avi")




if __name__ == "__main__":
    main()