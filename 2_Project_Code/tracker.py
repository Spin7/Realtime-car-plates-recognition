"""
tracker.py
----------
Temporal vehicle plate tracker with "coasting" support.

If a plate with confirmed text stops being detected, the tracker
keeps it visible for up to max_missed_frames frames with a progressive
fade-out (ghost track).

Key features:
- Association by Euclidean distance between centroids.
- Dynamic capture radius: shrinks linearly the longer a track
  goes without being detected (prevents a distant plate from
  wrongly claiming a ghost track).
- best_box: position saved at the moment of maximum OCR coherence,
  not at the last visible frame (which may be blurry).
- Majority vote over the OCR reading history to select confirmed_text.
"""

from collections import Counter, deque


class PlateTracker:
    """
    Parameters:
      history_len          -- OCR vote window per track
      dist_threshold       -- base radius (px) to associate a detection to an active track
      min_votes_to_confirm -- minimum votes to lock in confirmed_text and best_box
      max_missed_frames    -- frames without detection before discarding the track
      ghost_min_ratio      -- minimum fraction of the radius for expired ghost tracks
                             (e.g. 0.25 → at max_missed_frames the radius is 25%)
    """

    def __init__(self, history_len=12, dist_threshold=100,
                 min_votes_to_confirm=3, max_missed_frames=45,
                 ghost_min_ratio=0.25):
        self.history_len          = history_len
        self.dist_threshold       = dist_threshold
        self.min_votes_to_confirm = min_votes_to_confirm
        self.max_missed_frames    = max_missed_frames
        self.ghost_min_ratio      = ghost_min_ratio
        self.tracks               = {}
        self._next_id             = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _effective_threshold(self, missed_frames):
        """
        Capture radius based on frames without detection:
          missed=0   → full dist_threshold
          missed=max → dist_threshold * ghost_min_ratio
        """
        if missed_frames == 0:
            return self.dist_threshold
        ratio = 1.0 - (1.0 - self.ghost_min_ratio) * (
            min(missed_frames, self.max_missed_frames) / self.max_missed_frames
        )
        return self.dist_threshold * ratio

    def _find_track(self, cx, cy):
        """
        Finds the track closest to (cx, cy) within its effective radius.
        Ghost tracks have a reduced radius to avoid wrong claims.
        """
        best_id, best_dist = None, float("inf")
        for tid, data in self.tracks.items():
            tc, tr = data["center"]
            dist = ((cx - tc) ** 2 + (cy - tr) ** 2) ** 0.5
            thr  = self._effective_threshold(data["missed_frames"])
            if dist < thr and dist < best_dist:
                best_dist = dist
                best_id   = tid
        return best_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, cx, cy, text, box=None):
        """
        Registers an OCR reading for the detection centered at (cx, cy).

        Args:
            cx, cy : centroid of the detected bounding box
            text   : OCR text ("-" if nothing was recognized)
            box    : tuple (x1, y1, x2, y2) of the bounding box

        Returns:
            (text_to_display, is_confident: bool)
        """
        tid = self._find_track(cx, cy)
        if tid is None:
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {
                "center":          (cx, cy),
                "history":         deque(maxlen=self.history_len),
                "confirmed_text":  None,
                "confirmed_count": 0,
                "missed_frames":   0,
                "best_box":        box,
            }

        track = self.tracks[tid]
        track["missed_frames"] = 0   # seen this frame

        # Centroid with smooth moving average (70% previous, 30% new)
        oc, or_ = track["center"]
        track["center"] = (int(oc * 0.7 + cx * 0.3),
                           int(or_ * 0.7 + cy * 0.3))

        if text != "-":
            track["history"].append(text)

        history = track["history"]
        if not history:
            ct = track["confirmed_text"]
            return (ct, True) if ct else (text, False)

        counter = Counter(history)
        most_common, count = counter.most_common(1)[0]
        is_confident = count >= max(2, int(len(history) * 0.4))

        # Update confirmed_text and best_box only when votes improve
        if count >= self.min_votes_to_confirm:
            if track["confirmed_text"] != most_common or count > track["confirmed_count"]:
                track["confirmed_text"]  = most_common
                track["confirmed_count"] = count
                if box is not None:
                    track["best_box"] = box

        # Bad OCR but confirmed text exists → use it
        if text == "-" and track["confirmed_text"] is not None:
            return track["confirmed_text"], True

        return most_common, is_confident

    def tick_all(self, active_centers):
        """
        Called ONCE per frame with the centroids of current detections.
        Increments missed_frames on tracks that were not seen.
        """
        seen_ids = set()
        for cx, cy in active_centers:
            tid = self._find_track(cx, cy)
            if tid is not None:
                seen_ids.add(tid)
        for tid in self.tracks:
            if tid not in seen_ids:
                self.tracks[tid]["missed_frames"] += 1

    def get_ghost_tracks(self):
        """
        Returns tracks with confirmed text that have been missing for between
        1 and max_missed_frames frames (coasting zone).

        Each element is a dict with:
            center, best_box, confirmed_text, missed_frames
        """
        ghosts = []
        for data in self.tracks.values():
            if (data["confirmed_text"] is not None and
                    0 < data["missed_frames"] <= self.max_missed_frames):
                ghosts.append({
                    "center":         data["center"],
                    "best_box":       data["best_box"],
                    "confirmed_text": data["confirmed_text"],
                    "missed_frames":  data["missed_frames"],
                })
        return ghosts

    def cleanup_old_tracks(self):
        """Removes tracks that have exceeded max_missed_frames."""
        expired = [tid for tid, d in self.tracks.items()
                   if d["missed_frames"] > self.max_missed_frames]
        for tid in expired:
            del self.tracks[tid]
