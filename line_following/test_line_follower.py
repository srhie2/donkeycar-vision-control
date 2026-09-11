from types import SimpleNamespace

import cv2
import numpy as np
from simple_pid import PID

from line_following.line_follower import LineFollower


def make_config(**overrides):
    values = dict(
        OVERLAY_IMAGE=True,
        IMAGE_W=160,
        IMAGE_H=120,
        SCAN_Y=70,
        SCAN_HEIGHT=28,
        SCAN_EXTRA_ROWS=1,
        CV_INPUT_COLOR_ORDER="BGR",
        COLOR_THRESHOLD_LOW=(18, 18, 35),
        COLOR_THRESHOLD_HIGH=(35, 255, 255),
        COLOR_THRESHOLD_LOW_2=None,
        COLOR_THRESHOLD_HIGH_2=None,
        COLOR_DOMINANCE_MODE="YELLOW",
        COLOR_MIN_DOMINANCE=8,
        COLOR_MAX_CHANNEL_DIFF=30,
        TARGET_PIXEL=None,
        TARGET_THRESHOLD=10,
        CONFIDENCE_THRESHOLD=0.05,
        MAX_LINE_WIDTH_PX=25,
        MIN_LINE_ASPECT_RATIO=0.15,
        MIN_LINE_AREA_PX=6,
        MASK_MORPH_KERNEL_PX=1,
        MAX_LINE_JUMP_PX=25,
        ACQUIRE_MAX_DISTANCE_PX=30,
        REACQUIRE_LINE_AFTER_FRAMES=5,
        REACQUIRE_CONFIRM_FRAMES=1,
        REACQUIRE_CONFIRM_DISTANCE_PX=12,
        LINE_POSITION_SMOOTHING=0.65,
        THROTTLE_INITIAL=0.25,
        THROTTLE_STEP=0.02,
        THROTTLE_MAX=0.35,
        THROTTLE_MIN=0.25,
        NO_LINE_STOP_FRAMES=5,
        NO_LINE_THROTTLE_STEP=0.05,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def yellow_bgr():
    hsv = np.uint8([[[25, 180, 220]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


def make_controller(**overrides):
    pid = PID(-0.01, 0.0, -0.0001)
    return LineFollower(pid, make_config(**overrides))


def test_detects_a_centered_yellow_line():
    image = np.full((120, 160, 3), 90, dtype=np.uint8)
    image[70:98, 76:84] = yellow_bgr()

    line_x, confidence, mask = make_controller().get_i_color(image)

    assert 78 <= line_x <= 82
    assert confidence >= 0.9
    assert np.count_nonzero(mask) > 0


def test_rejects_a_far_initial_candidate():
    image = np.full((120, 160, 3), 90, dtype=np.uint8)
    image[70:98, 140:148] = yellow_bgr()

    line_x, confidence, _mask = make_controller().get_i_color(image)

    assert line_x == 0
    assert confidence == 0.0


def test_stops_when_no_line_is_detected():
    controller = make_controller()
    controller.steering = 0.4
    blank = np.full((120, 160, 3), 90, dtype=np.uint8)

    steering, throttle, _overlay = controller.run(blank)

    assert steering == 0.0
    assert throttle == 0.0


def test_debug_overlay_converts_bgr_to_rgb():
    image = np.full((120, 160, 3), (10, 20, 30), dtype=np.uint8)

    _steering, _throttle, overlay = make_controller().run(image)

    assert tuple(overlay[119, 159]) == (30, 20, 10)
