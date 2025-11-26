import numpy as np

class ShuttlecockInterpolator:
    def __init__(self):
        self.shuttlecock_id = 1

    def advanced_interpolation(self, detections, frame_width, frame_height):
        print("\n=== Shuttle interpolation ===")

        positions = []
        for detection in detections:
            if detection and self.shuttlecock_id in detection:
                bbox = detection[self.shuttlecock_id]
                positions.append(bbox)
            else:
                positions.append([])

        original_detections = sum(1 for pos in positions if pos)
        print(f"Model detection: {original_detections}/{len(positions)} frames")

        if original_detections == 0:
            print("Nothing to interpolate")
            return detections

        data = []
        for pos in positions:
            if pos:
                data.append(pos)
            else:
                data.append([np.nan, np.nan, np.nan, np.nan])

        arr = np.asarray(data, dtype=float) 

        segments = self.find_interpolation_segments(arr)
        print(f"Found {len(segments)} missing segments")

        arr_interpolated = arr.copy()

        for start_idx, end_idx in segments:
            segment_length = end_idx - start_idx + 1
            if 2 <= segment_length <= 20:
                before_start = max(0, start_idx - 1)
                after_end = min(len(arr_interpolated) - 1, end_idx + 1)
                if (not np.isnan(arr_interpolated[before_start, 0]) and
                    not np.isnan(arr_interpolated[after_end, 0])):
                    if segment_length <= 5:
                        self.linear_interpolation(arr_interpolated, start_idx, end_idx, before_start, after_end)
                    elif segment_length <= 15:
                        self.smooth_interpolation(arr_interpolated, start_idx, end_idx, before_start, after_end)
                    else:
                        self.cautious_interpolation(arr_interpolated, start_idx, end_idx, before_start, after_end)

        arr_poly = self.polynomial_interpolation(arr_interpolated)
        arr_filled = self.fill_nan(arr_poly)
        arr_smoothed = self.adaptive_smoothing(arr_filled)
        arr_final = self.velocity_limiting(arr_smoothed, frame_width, frame_height)

        final_detections = []
        for i in range(arr_final.shape[0]):
            x1 = arr_final[i, 0]
            if not np.isnan(x1):
                x1, y1, x2, y2 = arr_final[i].tolist()
                final_detections.append({self.shuttlecock_id: [x1, y1, x2, y2]})
            else:
                final_detections.append({})

        final_count = sum(1 for det in final_detections if det)
        improved_count = final_count - original_detections
        if original_detections > 0:
            print(f"Interpolated {improved_count} frames (+{improved_count / original_detections * 100:.1f}%)")

        return final_detections

    def find_interpolation_segments(self, arr):
        segments = []
        in_gap = False
        gap_start = 0
        for i in range(len(arr)):
            is_missing = np.isnan(arr[i, 0])
            if is_missing and not in_gap:
                in_gap = True
                gap_start = i
            elif not is_missing and in_gap:
                in_gap = False
                gap_end = i - 1
                if gap_end >= gap_start:
                    segments.append((gap_start, gap_end))
        if in_gap:
            segments.append((gap_start, len(arr) - 1))
        return segments

    def linear_interpolation(self, arr, start_idx, end_idx, before_start, after_end):
        for j in range(4):
            start_val = arr[before_start, j]
            end_val = arr[after_end, j]
            for i, current_idx in enumerate(range(start_idx, end_idx + 1)):
                t = (i + 1) / (end_idx - start_idx + 2)
                arr[current_idx, j] = start_val + (end_val - start_val) * t

    def smooth_interpolation(self, arr, start_idx, end_idx, before_start, after_end):
        for j in range(4):
            start_val = arr[before_start, j]
            end_val = arr[after_end, j]
            for i, current_idx in enumerate(range(start_idx, end_idx + 1)):
                t = (i + 1) / (end_idx - start_idx + 2)
                smooth_t = 0.5 - 0.5 * np.cos(t * np.pi)
                arr[current_idx, j] = start_val + (end_val - start_val) * smooth_t

    def cautious_interpolation(self, arr, start_idx, end_idx, before_start, after_end):
        for j in range(4):
            start_val = arr[before_start, j]
            end_val = arr[after_end, j]
            max_change = abs(end_val - start_val) * 0.8
            for i, current_idx in enumerate(range(start_idx, end_idx + 1)):
                t = (i + 1) / (end_idx - start_idx + 2)
                interpolated_val = start_val + (end_val - start_val) * t
                if i > 0:
                    prev_val = arr[current_idx - 1, j]
                    step = max_change / (end_idx - start_idx + 1) if (end_idx - start_idx + 1) > 0 else max_change
                    max_allowed = prev_val + step
                    min_allowed = prev_val - step
                    interpolated_val = np.clip(interpolated_val, min_allowed, max_allowed)
                arr[current_idx, j] = interpolated_val

    def polynomial_interpolation(self, arr):
        arr_poly = arr.copy()
        segments = self.find_continuous_segments(arr)
        for segment_start, segment_end in segments:
            segment_length = segment_end - segment_start + 1
            if segment_length >= 10:
                x = np.arange(segment_start, segment_end + 1)
                for j in range(4):
                    y = arr[segment_start:segment_end + 1, j]
                    valid_mask = ~np.isnan(y)
                    if np.sum(valid_mask) >= 5:
                        try:
                            x_valid = x[valid_mask]
                            y_valid = y[valid_mask]
                            coeffs = np.polyfit(x_valid, y_valid, 2)
                            poly = np.poly1d(coeffs)
                            for i_idx in range(segment_start, segment_end + 1):
                                if np.isnan(arr_poly[i_idx, j]):
                                    arr_poly[i_idx, j] = poly(i_idx)
                        except Exception:
                            pass
        return arr_poly

    def find_continuous_segments(self, arr):
        segments = []
        in_segment = False
        segment_start = 0
        for i in range(len(arr)):
            has_data = not np.isnan(arr[i, 0])
            if has_data and not in_segment:
                in_segment = True
                segment_start = i
            elif not has_data and in_segment:
                in_segment = False
                segment_end = i - 1
                if segment_end - segment_start >= 4:
                    segments.append((segment_start, segment_end))
        if in_segment and (len(arr) - 1 - segment_start >= 4):
            segments.append((segment_start, len(arr) - 1))
        return segments

    def fill_nan(self, arr):
        out = arr.copy()
        n, k = out.shape
        for j in range(k):
            last = np.nan
            for i in range(n):
                if np.isnan(out[i, j]):
                    out[i, j] = last
                else:
                    last = out[i, j]
            last = np.nan
            for i in range(n - 1, -1, -1):
                if np.isnan(out[i, j]):
                    out[i, j] = last
                else:
                    last = out[i, j]
        return out

    def adaptive_smoothing(self, arr):
        out = arr.copy()
        n, k = out.shape
        for j in range(k):
            col = out[:, j]
            diffs = np.abs(np.diff(col))
            diffs = diffs[~np.isnan(diffs)]
            if diffs.size == 0:
                continue
            avg_change = diffs.mean()
            if avg_change > 20:
                window = 3
            elif avg_change > 10:
                window = 5
            else:
                window = 7
            if window <= 1 or window > n:
                continue
            pad = window // 2
            col_pad = np.pad(col, (pad, pad), mode="edge")
            kernel = np.ones(window) / window
            smoothed = np.convolve(col_pad, kernel, mode="valid")
            out[:, j] = smoothed
        return out

    def velocity_limiting(self, arr, frame_width, frame_height):
        out = arr.copy()
        n, k = out.shape
        for i in range(1, n):
            if np.isnan(arr[i, 0]) or np.isnan(arr[i - 1, 0]):
                continue
            diff = np.abs(arr[i] - arr[i - 1])
            max_velocity = float(np.max(diff))
            if max_velocity > 100:
                factor = min(0.8, 100.0 / max_velocity)
                out[i] = arr[i - 1] + (arr[i] - arr[i - 1]) * factor
        return out
