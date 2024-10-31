from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_dist, get_center
class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def choose_and_filter_players(self, court_keypoints, player_detections):
        #only display player bounding boxes -> find IDs closest to keypoints => only based on first frame
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {trackID: bbox for trackID, bbox in player_dict.items() if trackID in chosen_players}
            filtered_player_detections.append(filtered_player_dict)  
            
        return filtered_player_detections 
    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center(bbox)
            
            min_dist = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                dist = measure_dist(player_center, court_keypoint)
                if dist < min_dist:
                    min_dist = dist        
            distances.append((track_id,min_dist)) 
        # sort to find 2 minimum ones
        distances.sort(key=lambda x: x[1])
        #choose first 2 players -> track IDs
        chosen_players = [distances[0][0],distances[1][0]]
        return chosen_players
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
            
        return player_detections    
    def detect_frame(self,frame):
        #persist = True -> for multiple frames
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        #key = player_ID, val = bounding box
        player_dict = {}
        for box in results.boxes:
            track_id =int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            obj_cls_id = box.cls.tolist()[0]
            obj_cls_name = id_name_dict[obj_cls_id]
            if obj_cls_name=="person":
                player_dict[track_id]=result
                
        return player_dict
    
    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
                
    