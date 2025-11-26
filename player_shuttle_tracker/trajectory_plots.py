import numpy as np
import matplotlib.pyplot as plt

def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def get_shuttle_centers(detections):
    idx = []
    xs = []
    ys = []
    for i, det in enumerate(detections):
        bbox = det.get(1) if det else None
        if bbox and len(bbox) == 4 and np.all(np.isfinite(bbox)):
            cx, cy = _bbox_center(bbox)
            idx.append(i)
            xs.append(float(cx))
            ys.append(float(cy))
    return np.asarray(idx), np.asarray(xs), np.asarray(ys)

def plot_xy_trajectory_colored_time(
    detections,
    out_path,
    frame_width,
    frame_height,
    invert_y=True,
    equal_aspect=True,
    cmap_name="viridis",
    point_size=6.0
):
    idx, xs, ys = get_shuttle_centers(detections)

    dpi = 100
    fig = plt.figure(figsize=(frame_width / dpi, frame_height / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111)

    ax.set_facecolor("white")
    ax.axis("off")

    if xs.size > 0:
        ax.scatter(xs, ys, c=idx, s=point_size, cmap=cmap_name)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if invert_y:
        ax.invert_yaxis()

    plt.savefig(out_path, dpi=dpi, bbox_inches=None, facecolor="white", pad_inches=0)
    plt.close()
