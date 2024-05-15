import pickle
import numpy as np
import os
import sys
sys.path.append("../")
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
import pandas as pd
from ultralytics import YOLO
import cv2
import supervision as sv

class Tracker:
    """
    A class used to track objects in video frames using a pre-trained YOLO model and ByteTrack for object tracking.

    Attributes:
    -----------
    model : YOLO
        An instance of the YOLO model used for object detection.
    tracker : ByteTrack
        An instance of the ByteTrack object tracker.
    """
    def __init__(self, model_path) -> None:
        """
        Initializes the Tracker class with a YOLO model.

        Parameters:
        -----------
        model_path : str
            The path to the pre-trained YOLO model.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def interpolate_ball_positions(self,ball_positions):
        """
        Interpolates missing ball positions in the given list of ball positions.

        Parameters:
        -----------
        ball_positions : list
            A list of dictionaries containing ball positions.

        Returns:
        --------
        list
            A list of dictionaries with interpolated ball positions.
        """

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        """
        Detects objects in the given list of video frames.

        Parameters:
        -----------
        frames : list
            A list of video frames (images).

        Returns:
        --------
        list
            A list of detection results for each frame.
        """

        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Gets object tracks from the given list of video frames. Can read from a stub file if provided.

        Parameters:
        -----------
        frames : list
            A list of video frames (images).
        read_from_stub : bool, optional
            Whether to read tracks from a stub file (default is False).
        stub_path : str, optional
            The path to the stub file (default is None).

        Returns:
        --------
        dict
            A dictionary containing tracks of players, referees, and the ball.
        """

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draws an ellipse around the object specified by the bounding box on the given frame.

        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame (image) on which to draw the ellipse.
        bbox : list
            The bounding box coordinates [x1, y1, x2, y2].
        color : tuple
            The color of the ellipse (BGR).
        track_id : int, optional
            The track ID of the object (default is None).

        Returns:
        --------
        numpy.ndarray
            The frame with the ellipse drawn on it.
        """

        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draws a triangle above the object specified by the bounding box on the given frame.

        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame (image) on which to draw the triangle.
        bbox : list
            The bounding box coordinates [x1, y1, x2, y2].
        color : tuple
            The color of the triangle (BGR).

        Returns:
        --------
        numpy.ndarray
            The frame with the triangle drawn on it.
        """

        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        """
        Draws annotations on the given video frames based on the object tracks.

        Parameters:
        -----------
        video_frames : list
            A list of video frames (images).
        tracks : dict
            A dictionary containing tracks of players, referees, and the ball.

        Returns:
        --------
        list
            A list of video frames with annotations drawn on them.
        """

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,0))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (143,47,206)) # Giving referee purple color

            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0,255,0))


            output_video_frames.append(frame)

        return output_video_frames
    