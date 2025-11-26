from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.player_class_id = 0 

    def detect_frame(self, frame, frame_num):
        results = self.model.predict(frame, conf=0.3, verbose=False)[0]
        
        players_dict = {}
        
        if results.boxes is not None and len(results.boxes) > 0:
            player_count = 0
            for box in results.boxes:
                class_id = int(box.cls.item())
                if class_id == self.player_class_id:  
                    bbox = box.xyxy.tolist()[0]
                    conf = box.conf.item()
                    player_count += 1
                    players_dict[player_count] = {
                        'bbox': bbox,
                        'confidence': conf
                    }
        
        return players_dict

    def detect_batch(self, frames):
        print("\n=== Player detection ===")
        print("Detection players ...")
        detections = []
        
        for i, frame in enumerate(frames):
            if i % 100 == 0:
                print(f" {i}/{len(frames)} frames")
            detections.append(self.detect_frame(frame, i))
        
        return detections