import numpy as np

class SimpleCentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # object_id: centroid
        self.max_distance = max_distance

    def update(self, detections):
        # detections: list of bounding boxes [x1, y1, x2, y2]
        if len(detections) == 0:
            return []
        input_centroids = np.array([
            [(x1 + x2) // 2, (y1 + y2) // 2] for (x1, y1, x2, y2) in detections
        ])
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values())) if self.objects else np.empty((0, 2))

        assignments = {}
        used_rows, used_cols = set(), set()

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.objects[self.next_object_id] = input_centroids[i]
                assignments[i] = self.next_object_id
                self.next_object_id += 1
            return [assignments[i] for i in range(len(input_centroids))]

        # Compute distances between new and existing centroids
        D = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            assignments[col] = object_id
            used_rows.add(row)
            used_cols.add(col)

        # Assign new IDs to unassigned detections
        for i in range(len(input_centroids)):
            if i not in assignments:
                self.objects[self.next_object_id] = input_centroids[i]
                assignments[i] = self.next_object_id
                self.next_object_id += 1

        return [assignments[i] for i in range(len(input_centroids))] 