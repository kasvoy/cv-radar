from ultralytics import YOLO
import cv2
import numpy as np
from speed_funcs import *

model = YOLO('yolov8l.pt')
video_path = "data/session5_center/video.avi"

def main():
    frame_number=0
    
    points = {
        'P1': [845, 60],
        'P2': [845, 277],
        'P3': [845, 502],
        'P4': [845, 731],
        'P5': [851, 956]
    }
    
    section_length = 7
    total_length = 28
    len25 = 21
    speeds_dict = dict()
    speed_limit = 80
    
    
    for result in model.track(source=video_path, show=False, stream=True, tracker='bytetrack.yaml', classes=[1, 2, 3, 5, 6, 7, 8]):
            
        frame = extract_roi(result.orig_img)
        boxes = result.boxes.xyxy.cpu().numpy()
        
        if result.boxes.id is not None:
            id_list = result.boxes.id.cpu().numpy().astype(int)
            track_tuples = (list(zip(id_list, boxes)))
        
        else:
            track_tuples = tuple()
          
        for track in track_tuples:
            track_id = track[0]
            bbox_coords = track[1]
            
            br = np.array([int(bbox_coords[2]), int(bbox_coords[3])], dtype=np.float32)
            br_transformed = get_point_under_transform(br, TRANSFORM_MATRIX)
            
            if track_id not in speeds_dict:
                speeds_dict[track_id] = {
                    'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'P5': 0,
                    'delta12': 0, 'delta23': 0, 'delta34': 0, 'delta45': 0, 'total_delta': 0,
                    'speed12': 0,'speed23': 0, 'speed34': 0, 'speed45': 0, 'avg_speed': 0}
            
            assign_point_frames(speeds_dict, points, br_transformed, track_id, frame_number, tolerance=10)
            
            avg_speed_kph = speeds_dict[track_id]['avg_speed']
            section_speeds = list(speeds_dict[track_id].values())[10:14]
            current_speed = section_speeds[0]
            
            if section_speeds[1] !=0 and section_speeds[2] == 0:
                current_speed= section_speeds[1]
                
            elif section_speeds[2] !=0 and section_speeds[3] == 0:
                current_speed= section_speeds[2]
            
            elif section_speeds[3] != 0:
                current_speed = section_speeds[3]
            
            point_frames = list(speeds_dict[track_id].values())[:5]
            
            if point_frames[0] !=0 and point_frames[4] !=0:
                total_delta = point_frames[4]/50 - point_frames[0]/50
                speeds_dict[track_id]['total_delta'] = total_delta
                
                speeds_dict[track_id]['avg_speed'] = round((total_length/total_delta) * 3.6, 3)
                
                text = f"average speed: {avg_speed_kph}"
                color_avg_speed = (255,255,255)
                
                if avg_speed_kph > speed_limit:
                    color_avg_speed = (0,0,255)
                
                cv2.putText(frame, text, (int(bbox_coords[0])+300, int(bbox_coords[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color_avg_speed, 2)
            
            
            
            txt = f"ID: {track_id}, speed: {current_speed}"
            bbox_color = (0,255,0)
            
            if current_speed > speed_limit:
                bbox_color = (0,0,255)
            
                
            
            cv2.rectangle(frame, (int(bbox_coords[0]), int(bbox_coords[1])), (int(bbox_coords[2]), int(bbox_coords[3])), bbox_color, 1)
            cv2.putText(frame, txt, (int(bbox_coords[0]), int(bbox_coords[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)


        cv2.imshow("yolov8", frame)
        frame_number += 1
        
        key = cv2.waitKey(1)
                
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('p'):
            cv2.waitKey(-1)

if __name__ == '__main__':
    main()
