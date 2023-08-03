from strongsort.strong_sort import StrongSORT
#from strongsort import StrongSORT
from ultralytics import YOLO
import cv2
from speed_funcs import *
import torch
from pathlib import Path

model = YOLO('yolov8m.pt')
video_path = "data/session5_center/video.avi"

tracker = StrongSORT(model_weights=Path('osnet_x0_25_msmt17.pt'), device=torch.device('cuda'),fp16=False, ema_alpha=0.8962, max_age=5, max_dist=0.16, max_iou_distance=0.54, mc_lambda=0.995, n_init=3, nn_budget=100)

def main():
    
    points = {
    'P1': [845, 60],
    'P2': [845, 277],
    'P3': [845, 502],
    'P4': [845, 731],
    'P5': [851, 956]
    }
    total_length = 28
    speeds_dict = dict()
    speed_limit = 80
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC,94*1000)
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = extract_roi(frame)
        
        if ret:
            results = model(frame)
            
            for result in results:
                boxes = result[:].boxes.xyxy
                score = result[:].boxes.conf
                category_id = result[:].boxes.cls
                dets = torch.cat((boxes, score.unsqueeze(1), category_id.unsqueeze(1)), dim=1)
                tracks = tracker.update(dets.cpu(), frame)

                for track in tracks:
                    bbox, track_id, category_id, score = (
                        track[:4],
                        int(track[4]),
                        track[5],
                        track[6],
                    )
                    #cv2.putText(frame, f"{track_id}", (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 1)
                    
                    br = np.array([int(bbox[2]), int(bbox[3])], dtype=np.float32)
                    br_transformed = get_point_under_transform(br, TRANSFORM_MATRIX)
                    
                    if track_id not in speeds_dict:
                        speeds_dict[track_id] = {
                            'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'P5': 0,
                            'delta12': 0, 'delta23': 0, 'delta34': 0, 'delta45': 0, 'total_delta': 0,
                            'speed12': 0,'speed23': 0, 'speed34': 0, 'speed45': 0, 'avg_speed': 0}
                    
                    assign_point_time(speeds_dict, points, br_transformed, track_id, current_time, tolerance=10)
                    
                    avg_speed_kph = speeds_dict[track_id]['avg_speed']
                    section_speeds = list(speeds_dict[track_id].values())[10:14]
                    current_speed = section_speeds[0]
                    
                    if section_speeds[1] !=0 and section_speeds[2] == 0:
                        current_speed= section_speeds[1]
                        
                    elif section_speeds[2] !=0 and section_speeds[3] == 0:
                        current_speed= section_speeds[2]
                    
                    elif section_speeds[3] != 0:
                        current_speed = section_speeds[3]
                    
                    point_times = list(speeds_dict[track_id].values())[:5]
                    
                    if point_times[0] !=0 and point_times[4] !=0:
                        total_delta = point_times[4] - point_times[0]
                        speeds_dict[track_id]['total_delta'] = total_delta
                        
                        speeds_dict[track_id]['avg_speed'] = round((total_length/total_delta) * 3.6, 3)
                        
                        text = f"average speed: {avg_speed_kph}"
                        color_avg_speed = (255,255,255)
                        
                        if avg_speed_kph > speed_limit:
                            color_avg_speed = (0,0,255)
                        
                        cv2.putText(frame, text, (int(bbox[0]) + 300, int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color_avg_speed, 2)
                    
                    
                    txt = f"ID: {track_id}, speed: {current_speed}"
                    bbox_color = (0,255,0)
                    
                    if current_speed > speed_limit:
                        bbox_color = (0,0,255)
                    
                    
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), bbox_color, 1)
                    cv2.putText(frame, txt, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    
            current_time = float('{:.2f}'.format(cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
            display_video_time(current_time, frame)
            
            cv2.imshow('Video', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('p'):
                cv2.waitKey(-1)
            
        else:
            break

    
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()