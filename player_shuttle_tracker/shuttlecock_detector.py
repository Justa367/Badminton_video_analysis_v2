# player_shuttle_tracker/shuttlecock_detector.py
from ultralytics import YOLO

class ShuttlecockDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.shuttlecock_id = 1

    def detect_frame(self, frame, frame_num):
        results = self.model.predict(frame, conf=0.25, verbose=False)[0]
        
        shuttlecock_dict = {}
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls.item())
                if class_id == 0:
                    bbox = box.xyxy.tolist()[0]
                    shuttlecock_dict[self.shuttlecock_id] = bbox
                    break
        
        return shuttlecock_dict

    def detect_batch(self, frames):
        print("\n=== Shuttle detection ===")
        print("Tracking shuttle ...")
        detections = []
        
        for i, frame in enumerate(frames):
            if i % 100 == 0:
                print(f"{i}/{len(frames)} frames")
            detections.append(self.detect_frame(frame, i))
        
        return detections

    def is_shuttlecock_on_screen(self, bbox, frame_width, frame_height):
        if not bbox or len(bbox) != 4:
            return False
            
        x1, y1, x2, y2 = bbox
        margin = 50
        on_screen = (x1 > -margin and x2 < frame_width + margin and 
                    y1 > -margin and y2 < frame_height + margin)
        return on_screen