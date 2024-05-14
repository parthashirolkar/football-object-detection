from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            break
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)


        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }


        for frame_num, detection in enumerate(detections):
            class_names = detections.names
            class_names_inversed = {v:k for k, v in class_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] == class_names_inversed['person']
            
            
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inversed['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if class_id == class_names_inversed['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names_inversed['ball']:
                    tracks['ball'][frame_num][1] = {"bbox": bbox}

            print(detections_with_tracks)

