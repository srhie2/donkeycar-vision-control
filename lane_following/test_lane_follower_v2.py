import csv
import os
import tempfile
import time
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from simple_pid import PID

from lane_following.lane_follower_v2 import LaneFollower


def hsv_rgb(h, s, v):
    return tuple(
        int(x) for x in cv2.cvtColor(
            np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB
        )[0, 0]
    )


BLUE = hsv_rgb(90, 150, 180)
WHITE = hsv_rgb(80, 10, 200)


def config(**overrides):
    values = dict(
        CAR_PATH="/tmp",
        OVERLAY_IMAGE=True,
        LANE_FOLLOW_DEBUG=False,
        LANE_V2_CSV_LOG=False,
        LANE_V2_ROI_TOP=0.0,
        LANE_V2_PERSPECTIVE_POINTS=((0, 1), (0, 0), (1, 0), (1, 1)),
        YELLOW_THRESHOLD_LOW=(80, 50, 100),
        YELLOW_THRESHOLD_HIGH=(105, 220, 210),
        YELLOW2_THRESHOLD_LOW=(0, 0, 0),
        YELLOW2_THRESHOLD_HIGH=(0, 0, 0),
        WHITE_THRESHOLD_LOW=(50, 0, 150),
        WHITE_THRESHOLD_HIGH=(110, 40, 220),
        LANE_V2_MORPH_KERNEL=3,
        LANE_V2_MEDIAN_KERNEL=0,
        LANE_V2_MIN_BLOB_AREA=3,
        LANE_V2_MAX_BLOB_AREA=5000,
        LANE_V2_MIN_BLOB_HEIGHT=2,
        LANE_V2_MAX_BLOB_ASPECT=8,
        LANE_V2_MIN_FIT_PIXELS=10,
        LANE_V2_MIN_FIT_Y_SPAN=0.2,
        LANE_V2_MAX_FIT_RESIDUAL=15,
        LANE_V2_EXPECTED_LANE_WIDTH_PX=80,
        LANE_V2_LANE_WIDTH_TOLERANCE=0.25,
        LANE_V2_LOOKAHEAD=0.68,
        LANE_V2_EMA_ALPHA=1.0,
        LANE_V2_RECOVERY_TIMEOUT_S=0.25,
        LANE_V2_SUPPORT_PIXELS=100,
        LANE_V2_TEMPORAL_SCALE_PX=30,
        LANE_V2_CONFIDENCE_STOP=0.2,
        LANE_V2_CONFIDENCE_CRAWL=0.5,
        LANE_V2_CONFIDENCE_NORMAL=0.8,
        LANE_V2_PID_OUTPUT_LIMIT=1.0,
        LANE_V2_STEERING_SIGN=1.0,
        LANE_V2_STEERING_RATE_LIMIT=0.08,
        LANE_V2_THROTTLE_NORMAL=0.45,
        LANE_V2_THROTTLE_MEDIUM=0.35,
        LANE_V2_THROTTLE_TURN=0.25,
        LANE_V2_THROTTLE_CRAWL=0.15,
        LANE_V2_THROTTLE_ACCEL_LIMIT=1.0,
        LANE_V2_THROTTLE_DECEL_LIMIT=1.0,
        LANE_V2_TURN_STEERING=0.35,
        LANE_V2_SHARP_STEERING=0.65,
        LANE_V2_CURVATURE_SLOW=0.2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def lane_image(left_bottom=40, right_bottom=120, offset=0, curved=False,
               left=True, right=True, yellow_on_right=False):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    ys = np.arange(120)
    bend = 0.0018 * np.square(ys - 60) if curved else np.zeros_like(ys)
    left_x = left_bottom + offset + bend + (119 - ys) * 10 / 119
    right_x = right_bottom + offset + bend + (119 - ys) * -10 / 119
    left_color, right_color = (
        (WHITE, BLUE) if yellow_on_right else (BLUE, WHITE)
    )
    if left:
        points = np.column_stack((left_x.astype(np.int32), ys)).reshape((-1, 1, 2))
        cv2.polylines(image, [points], False, left_color, 5)
    if right:
        points = np.column_stack((right_x.astype(np.int32), ys)).reshape((-1, 1, 2))
        cv2.polylines(image, [points], False, right_color, 5)
    return image


class LaneFollowerV2Tests(unittest.TestCase):
    def follower(self, **overrides):
        return LaneFollower(PID(0.01, 0.0, 0.0001), config(**overrides))

    def test_dual_boundaries_produce_center_and_high_confidence(self):
        follower = self.follower()
        steering, throttle, overlay = follower.run(lane_image())
        self.assertEqual(follower._dbg["state"], "dual")
        self.assertAlmostEqual(follower._dbg["lane_center"], 80, delta=2)
        self.assertGreater(follower._dbg["confidence"], 0.8)
        self.assertAlmostEqual(steering, 0, delta=0.03)
        self.assertEqual(throttle, 0.45)
        self.assertEqual(overlay.shape, (120, 160, 3))
        self.assertEqual(follower._dbg["lane_layout"], "right_lane")

    def test_yellow_on_right_selects_left_lane(self):
        follower = self.follower()
        steering, _, _ = follower.run(lane_image(yellow_on_right=True))
        self.assertEqual(follower._dbg["state"], "dual")
        self.assertEqual(follower._dbg["lane_layout"], "left_lane")
        self.assertAlmostEqual(follower._dbg["lane_center"], 80, delta=2)
        self.assertAlmostEqual(steering, 0, delta=0.03)

    def test_curved_offset_lane_steers_with_rate_limit(self):
        follower = self.follower()
        steering, _, _ = follower.run(lane_image(offset=20, curved=True))
        self.assertGreater(steering, 0)
        self.assertLessEqual(abs(steering), 0.08001)
        self.assertGreater(follower._dbg["curvature"], 0)

    def test_single_boundary_is_inferred(self):
        follower = self.follower()
        _, throttle, _ = follower.run(lane_image(right=False))
        self.assertEqual(follower._dbg["state"], "single_inferred")
        self.assertGreaterEqual(follower._dbg["confidence"], 0.5)
        self.assertLessEqual(follower._dbg["confidence"], 0.72)
        self.assertEqual(throttle, 0)

    def test_single_boundary_can_crawl_when_enabled(self):
        follower = self.follower(LANE_V2_ALLOW_SINGLE_BOUNDARY=True)
        _, throttle, _ = follower.run(lane_image(right=False))
        self.assertEqual(follower._dbg["state"], "single_inferred")
        self.assertEqual(throttle, 0.15)

    def test_ransac_ignores_large_white_noise_blob(self):
        follower = self.follower()
        image = lane_image()
        image[40:80, 60:95] = WHITE  # glare-like blob between the two markings
        follower.run(image)
        self.assertEqual(follower._dbg["state"], "dual")
        self.assertAlmostEqual(follower._dbg["lane_center"], 80, delta=6)

    def test_wide_blob_is_rejected_by_width_filter(self):
        follower = self.follower(LANE_V2_MAX_BLOB_WIDTH_FRAC=0.35)
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[30:110, 20:140] = WHITE  # 120px-wide wall-like region
        follower.run(image)
        self.assertEqual(follower._dbg["state"], "lost")
        self.assertEqual(follower.throttle, 0)

    def test_impossible_width_falls_back_to_single_boundary(self):
        follower = self.follower()
        follower.run(lane_image(left_bottom=65, right_bottom=95))
        self.assertEqual(follower._dbg["state"], "single_inferred")
        # movement stays blocked unless single-boundary crawl is enabled
        self.assertEqual(follower.throttle, 0)

    def test_temporal_hold_then_timeout_stops(self):
        follower = self.follower()
        follower.run(lane_image())
        follower.run(np.zeros((120, 160, 3), dtype=np.uint8))
        self.assertEqual(follower._dbg["state"], "holding")
        self.assertEqual(follower.throttle, 0)
        follower._last_detection_time = time.monotonic() - 1.0
        follower.run(np.zeros((120, 160, 3), dtype=np.uint8))
        self.assertEqual(follower._dbg["state"], "timeout")
        self.assertEqual(follower.throttle, 0)

    def test_missing_image_stops_and_clears_steering(self):
        follower = self.follower()
        follower.run(lane_image(offset=20))
        steering, throttle, overlay = follower.run(None)
        self.assertEqual((steering, throttle, overlay), (0.0, 0.0, None))

    def test_live_hsv_update_changes_thresholds(self):
        follower = self.follower()
        follower.apply_hsv_ranges(((1, 2, 3), (4, 5, 6)), ((7, 8, 9), (10, 11, 12)))
        self.assertEqual(tuple(follower.yellow_lo), (1, 2, 3))
        self.assertEqual(tuple(follower.white_hi), (10, 11, 12))

    def test_runtime_resolution_scales_pixel_and_area_thresholds(self):
        follower = self.follower()
        large = cv2.resize(lane_image(), (320, 240), interpolation=cv2.INTER_NEAREST)
        follower.run(large)
        self.assertEqual(follower._dbg["state"], "dual")
        self.assertAlmostEqual(follower.expected_lane_width, 160, delta=0.1)
        self.assertAlmostEqual(follower._dbg["lane_center"], 160, delta=4)

    def test_curvature_feedforward_anticipates_far_bend(self):
        # Straight/centered near the car (y >= 60, matching the default
        # lane_image() baseline exactly); bends right only further out
        # (y < 60, since warped y=0 is the horizon), so the near-lookahead
        # error stays ~0 while the far-lookahead preview error is positive.
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        ys = np.arange(120)
        shift = np.where(ys < 60, 30.0 * (60 - ys) / 60.0, 0.0)
        left_pts = np.column_stack(((40 + shift).astype(np.int32), ys)).reshape((-1, 1, 2))
        right_pts = np.column_stack(((120 + shift).astype(np.int32), ys)).reshape((-1, 1, 2))
        cv2.polylines(image, [left_pts], False, BLUE, 5)
        cv2.polylines(image, [right_pts], False, WHITE, 5)

        baseline = self.follower(LANE_V2_CURVATURE_FF_GAIN=0.0)
        steering_no_ff, _, _ = baseline.run(image)
        self.assertAlmostEqual(steering_no_ff, 0.0, delta=0.01)

        with_ff = self.follower()  # default LANE_V2_CURVATURE_FF_GAIN
        steering_with_ff, _, _ = with_ff.run(image)
        self.assertGreater(with_ff._dbg["error"], -1)
        self.assertAlmostEqual(with_ff._dbg["error"], 0.0, delta=1.0)
        self.assertGreater(steering_with_ff, steering_no_ff)

    def test_csv_has_stable_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "lane.csv")
            follower = self.follower(LANE_V2_CSV_LOG=True, LANE_V2_CSV_LOG_PATH=path)
            follower.run(lane_image())
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(tuple(rows[0]), LaneFollower.CSV_FIELDS)
            self.assertEqual(rows[0]["state"], "dual")


if __name__ == "__main__":
    unittest.main()
