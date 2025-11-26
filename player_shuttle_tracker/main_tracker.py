import cv2
import os
import sys
import numpy as np

from player_shuttle_tracker.shuttlecock_detector import ShuttlecockDetector
from player_shuttle_tracker.shuttlecock_interpolator import ShuttlecockInterpolator
from player_shuttle_tracker.player_detector import PlayerDetector
from player_shuttle_tracker.court_detection import (
    CourtPartDetector,
    choose_best_annotation_set,
    draw_annotations,
)
from player_shuttle_tracker.trajectory_plots import plot_xy_trajectory_colored_time


def make_output_dirs(input_video_path: str):
    base = os.path.splitext(os.path.basename(input_video_path))[0]
    out_root = os.path.join("player_shuttle_tracker", "output_vid")
    vid_dir = os.path.join(out_root, base)
    stats_dir = os.path.join(vid_dir, "stats")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    return out_root, vid_dir, stats_dir


class DualTracker:
    def __init__(self, shuttlecock_model_path, player_model_path, court_model_path=None,
                 court_conf=0.25, court_warmup_frames=50):
        self.shuttlecock_detector = ShuttlecockDetector(shuttlecock_model_path)
        self.shuttlecock_interpolator = ShuttlecockInterpolator()
        self.player_detector = PlayerDetector(player_model_path)
        self.court_model_path = court_model_path
        self.court_conf = court_conf
        self.court_warmup_frames = court_warmup_frames
        self.static_court_ann = []

    def process_video(self, input_video_path, output_video_path=None):
        frames = self.load_video_frames(input_video_path)
        if len(frames) == 0:
            return

        out_root, vid_dir, stats_dir = make_output_dirs(input_video_path)
        if output_video_path is None:
            output_video_path = os.path.join(vid_dir, "output.mp4")

        if self.court_model_path:
            self.prepare_static_court_annotations(frames)

        raw_shuttlecock_detections = self.shuttlecock_detector.detect_batch(frames)

        frame_height, frame_width = frames[0].shape[:2]
        final_shuttlecock_detections = self.shuttlecock_interpolator.advanced_interpolation(
            raw_shuttlecock_detections, frame_width, frame_height
        )

        player_detections = self.player_detector.detect_batch(frames)

        traj_path = os.path.join(stats_dir, "trajectory_xy.png")
        plot_xy_trajectory_colored_time(
            final_shuttlecock_detections,
            out_path=traj_path,
            frame_width=frame_width,
            frame_height=frame_height,
            invert_y=True,
            equal_aspect=True,
            cmap_name="viridis",
            point_size=6.0
        )

        self.save_output_video(
            frames, final_shuttlecock_detections, player_detections,
            input_video_path, output_video_path
        )

    def prepare_static_court_annotations(self, frames):
        if not isinstance(self.court_model_path, str) or not os.path.exists(self.court_model_path):
            self.static_court_ann = []
            return

        warmup_N = min(self.court_warmup_frames, len(frames))

        try:
            court_det = CourtPartDetector(self.court_model_path, conf=self.court_conf)
        except Exception:
            self.static_court_ann = []
            return

        annotations_per_frame = []
        for i in range(warmup_N):
            try:
                ann = court_det.detect_frame(frames[i])
            except Exception:
                ann = []
            annotations_per_frame.append(ann)

        try:
            best_idx, best_ann = choose_best_annotation_set(annotations_per_frame)
        except Exception:
            self.static_court_ann = []
            return

        if best_idx is None or len(best_ann) == 0:
            self.static_court_ann = []
        else:
            self.static_court_ann = best_ann

    def load_video_frames(self, input_video_path):
        cap = cv2.VideoCapture(input_video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def draw_tracking(self, frame, shuttlecock_dict, players_dict, frame_idx):
        frame_copy = frame.copy()
        frame_height, frame_width = frame_copy.shape[:2]

        if self.static_court_ann:
            frame_copy = draw_annotations(frame_copy, self.static_court_ann, alpha_mask=0.35)

        if (shuttlecock_dict and self.shuttlecock_detector.shuttlecock_id in shuttlecock_dict and
                self.shuttlecock_detector.is_shuttlecock_on_screen(
                    shuttlecock_dict[self.shuttlecock_detector.shuttlecock_id], frame_width, frame_height)):
            bbox = shuttlecock_dict[self.shuttlecock_detector.shuttlecock_id]
            try:
                bx = np.array(bbox, dtype=np.float64)
                x1, y1, x2, y2 = map(int, bx.tolist())
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame_copy, "SHUTTLECOCK", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception:
                pass

        for player_id, player_data in players_dict.items():
            bbox = player_data.get('bbox')
            conf = player_data.get('confidence')
            if bbox is None:
                continue
            try:
                bx = np.array(bbox, dtype=np.float64)
                x1, y1, x2, y2 = map(int, bx.tolist())
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"PLAYER {player_id} ({conf:.2f})" if conf is not None else f"PLAYER {player_id}"
                cv2.putText(frame_copy, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            except Exception:
                continue

        cv2.putText(frame_copy, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame_copy

    def save_output_video(self, frames, shuttlecock_detections, player_detections,
                          input_video_path, output_video_path):
        if not frames:
            return

        height, width = frames[0].shape[:2]
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        try:
            if fps is None or fps <= 0 or not np.isfinite(fps):
                fps = 25.0
        except Exception:
            fps = 25.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (width, height))

        for i, (frame, shuttlecock_dict, players_dict) in enumerate(
                zip(frames, shuttlecock_detections, player_detections)):
            output_frame = self.draw_tracking(frame, shuttlecock_dict, players_dict, i)
            out.write(output_frame)

        out.release()


def main():
        #shuttlecock_model_path = r"player_shuttle_tracker/comparasion/best/yolov8s_best.pt" #player_shuttle_tracker/models/shuttle_detection.pt
        shuttlecock_model_path = r"models/shuttle_detection.pt"
        player_model_path      = r"models/player_detection.pt" 
        court_model_path       = r"models/court_detection.pt" 
        input_video            = r"chopped_videos/clip_7.mp4"

        os.makedirs("player_shuttle_tracker", exist_ok=True)

        tracker = DualTracker(
            shuttlecock_model_path=shuttlecock_model_path,
            player_model_path=player_model_path,
            court_model_path=court_model_path,
            court_conf=0.25,
            court_warmup_frames=50
        )

        tracker.process_video(input_video)


if __name__ == "__main__":
    main()
