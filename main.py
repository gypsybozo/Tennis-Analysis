from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import courtLineDetector
from mini_court import MiniCourt
from utils import measure_dist, convert_pixel_distance_to_meters,draw_player_stats
import constants
import cv2
from copy import deepcopy
import pandas as pd
def main():
    #read in the vid
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    #detecting players, ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolov5_last.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    #courtline detector
    court_model_path = "models/keypoints_model_batch_8_threads_1.pth"
    court_line_detector = courtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #only display player bounding boxes -> find IDs closest to keypoints
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    
    # mini court
    mini_court = MiniCourt(video_frames[0])
    
    # detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    # print(ball_shot_frames)
    
    #convert positions of player and ball to minicourt
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                           ball_detections,
                                                                                                                           court_keypoints)
    player_stats_data = [
        {
            'frame_num':0,
            'player_1_num_of_shots':0,
            'player_1_total_shot_speed':0,
            'player_1_last_shot_speed':0,
            'player_1_total_player_speed':0,
            'player_1_last_player_speed':0,
            
            'player_2_num_of_shots':0,
            'player_2_total_shot_speed':0,
            'player_2_last_shot_speed':0,
            'player_2_total_player_speed':0,
            'player_2_last_player_speed':0,
        }
    ]
    for ball_shot_id in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_id]
        end_frame = ball_shot_frames[ball_shot_id+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 #24fps
        #dist covered by ball
        dist_covered_by_ball_in_pixels = measure_dist(ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])
        dist_covered_by_ball_in_meters = convert_pixel_distance_to_meters( dist_covered_by_ball_in_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 
        #speed of shot in kmph
        speed_of_ball_shot = dist_covered_by_ball_in_meters/ball_shot_time_in_seconds * 3.6
        
        #which player hit the shot
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_dist(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))
        #speed of opponent 
        opponent_player_id = 1 if player_shot_ball==2 else 2
        dist_covered_by_opp_in_pixels = measure_dist(player_mini_court_detections[start_frame][opponent_player_id],
                                                     player_mini_court_detections[end_frame][opponent_player_id])
        dist_covered_by_opp_in_meters = convert_pixel_distance_to_meters(dist_covered_by_opp_in_pixels,
                                                                         constants.DOUBLE_LINE_WIDTH,
                                                                         mini_court.get_width_of_mini_court())
        speed_of_opponent = dist_covered_by_opp_in_meters/ball_shot_time_in_seconds * 3.6
        
        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_num_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)
        
    
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num':list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num',how='left')
    #reps NA w prev not NA
    player_stats_data_df = player_stats_data_df.ffill()
    
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_num_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_num_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_num_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_num_of_shots']
        
        
    #Draw output
    # Draw player bboxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    # draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    
    #draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections,color=(0,255,255))
    
    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)
    
    # draw frame num
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    #saving video
    save_video(output_video_frames , "/Users/kriti.bharadwaj03/Desktop/tennis_analysis/output_videos/output_video.avi")
if __name__=="__main__":
    main()