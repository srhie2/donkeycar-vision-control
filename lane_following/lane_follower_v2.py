"""Production-style classical lane estimation for DonkeyCar.

The public Donkey part interface intentionally remains compatible with the
original follower: ``LaneFollower(pid, cfg)`` and ``run(rgb_image)``.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np

logger = logging.getLogger(__name__)
_FILE_HANDLER_READY = False


def _ensure_file_logger(log_path):
    global _FILE_HANDLER_READY
    if _FILE_HANDLER_READY:
        return
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        handler = RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        _FILE_HANDLER_READY = True
        logger.info("LaneFollower v2 debug log -> %s", log_path)
    except OSError as exc:
        logger.warning("Could not open lane follow log %s: %s", log_path, exc)


def _odd(value, minimum=1):
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _ema(previous, current, alpha):
    if current is None:
        return previous
    if previous is None:
        return float(current)
    return float(alpha * current + (1.0 - alpha) * previous)


@dataclass
class BoundaryFit:
    coeffs: np.ndarray
    degree: int
    residual: float
    pixels: int
    y_span: float
    inferred: bool = False

    def x(self, y):
        return np.polyval(self.coeffs, y)


class LaneFollower:
    CSV_FIELDS = (
        "timestamp", "frame", "lane_center", "image_center", "offset",
        "confidence", "lane_width", "left_detected", "right_detected",
        "pid_p", "pid_i", "pid_d", "steering", "throttle", "fps",
        "latency_ms", "state",
    )

    def __init__(self, pid, cfg):
        self.pid_st = pid
        self.overlay_image = bool(getattr(cfg, "OVERLAY_IMAGE", True))

        # The OAK-D delivers BGR frames, but this part historically ran
        # cv2.COLOR_RGB2HSV on them, swapping red<->blue and pushing the blue
        # divider's hue from ~112 to ~8 so no HSV range could ever match it
        # (Sungsan's line follower hit the same issue; see
        # experiments/sungsan_line_following). Select the conversion from the
        # true input order. Default "RGB" keeps synthetic-image unit tests
        # unchanged; the real car sets LANE_V2_INPUT_COLOR_ORDER = "BGR".
        order = str(getattr(cfg, "LANE_V2_INPUT_COLOR_ORDER", "RGB")).upper()
        self.hsv_code = cv2.COLOR_BGR2HSV if order == "BGR" else cv2.COLOR_RGB2HSV

        # Existing names are retained because the live calibrator uses them.
        self.yellow_lo = np.asarray(
            getattr(cfg, "YELLOW_THRESHOLD_LOW", (80, 50, 100)), dtype=np.uint8
        )
        self.yellow_hi = np.asarray(
            getattr(cfg, "YELLOW_THRESHOLD_HIGH", (105, 220, 210)), dtype=np.uint8
        )
        self.yellow2_lo = np.asarray(
            getattr(cfg, "YELLOW2_THRESHOLD_LOW", (0, 0, 0)), dtype=np.uint8
        )
        self.yellow2_hi = np.asarray(
            getattr(cfg, "YELLOW2_THRESHOLD_HIGH", (0, 0, 0)), dtype=np.uint8
        )
        self.white_lo = np.asarray(
            getattr(cfg, "WHITE_THRESHOLD_LOW", (0, 0, 150)), dtype=np.uint8
        )
        self.white_hi = np.asarray(
            getattr(cfg, "WHITE_THRESHOLD_HIGH", (179, 50, 255)), dtype=np.uint8
        )
        # White/edge detection mode. "hsv" = the original absolute-brightness
        # inRange (kept as default so synthetic unit tests are unchanged).
        # "tophat" = local-contrast bright-ridge detector: on real sunlit
        # pavement the road is as bright as the painted edge line (both V~220+,
        # low saturation), so an absolute V floor floods the whole road. A
        # morphological top-hat keeps only pixels brighter than their local
        # surroundings -- the curb/edge line survives, the uniformly-bright
        # road does not. Gated to low saturation so the coloured yellow/blue
        # markings never leak into the edge mask.
        self.white_mode = str(getattr(cfg, "LANE_V2_WHITE_MODE", "hsv")).lower()
        self.white_tophat_kernel = _odd(
            getattr(cfg, "LANE_V2_WHITE_TOPHAT_KERNEL", 15), 3
        )
        self.white_tophat_thresh = int(
            getattr(cfg, "LANE_V2_WHITE_TOPHAT_THRESH", 35)
        )
        self.white_max_sat = int(getattr(cfg, "LANE_V2_WHITE_MAX_SAT", 60))

        self.roi_top = float(np.clip(getattr(cfg, "LANE_V2_ROI_TOP", 0.45), 0.0, 0.9))
        self.perspective_points = np.asarray(getattr(
            cfg,
            "LANE_V2_PERSPECTIVE_POINTS",
            ((0.08, 0.98), (0.34, 0.05), (0.66, 0.05), (0.92, 0.98)),
        ), dtype=np.float32)
        if self.perspective_points.shape != (4, 2):
            raise ValueError("LANE_V2_PERSPECTIVE_POINTS must contain four (x, y) pairs")

        self.morph_kernel = _odd(getattr(cfg, "LANE_V2_MORPH_KERNEL", 3), 1)
        self.median_kernel = int(getattr(cfg, "LANE_V2_MEDIAN_KERNEL", 0))
        self.min_blob_area = int(getattr(cfg, "LANE_V2_MIN_BLOB_AREA", 8))
        self.max_blob_area = int(getattr(cfg, "LANE_V2_MAX_BLOB_AREA", 5000))
        self.min_blob_height = int(getattr(cfg, "LANE_V2_MIN_BLOB_HEIGHT", 3))
        self.max_blob_aspect = float(getattr(cfg, "LANE_V2_MAX_BLOB_ASPECT", 6.0))
        self.min_fit_pixels = int(getattr(cfg, "LANE_V2_MIN_FIT_PIXELS", 18))
        self.min_fit_y_span = float(getattr(cfg, "LANE_V2_MIN_FIT_Y_SPAN", 0.18))
        self.max_fit_residual = float(getattr(cfg, "LANE_V2_MAX_FIT_RESIDUAL", 12.0))
        self.max_blob_width_frac = float(
            getattr(cfg, "LANE_V2_MAX_BLOB_WIDTH_FRAC", 0.40)
        )
        self.ransac_enabled = bool(getattr(cfg, "LANE_V2_RANSAC_ENABLED", True))
        self.ransac_threshold = float(getattr(cfg, "LANE_V2_RANSAC_THRESHOLD_PX", 4.0))
        self.ransac_iterations = int(getattr(cfg, "LANE_V2_RANSAC_ITERATIONS", 40))
        self._rng = np.random.default_rng(20260717)

        self.reference_width = float(getattr(cfg, "LANE_V2_REFERENCE_WIDTH", 160))
        self.reference_height = float(getattr(cfg, "LANE_V2_REFERENCE_HEIGHT", 120))
        self.expected_lane_width_cfg = float(
            getattr(cfg, "LANE_V2_EXPECTED_LANE_WIDTH_PX", 100)
        )
        self.expected_lane_width = self.expected_lane_width_cfg
        self.width_tolerance = float(getattr(cfg, "LANE_V2_LANE_WIDTH_TOLERANCE", 0.25))
        self.lookahead = float(np.clip(getattr(cfg, "LANE_V2_LOOKAHEAD", 0.68), 0.05, 0.98))
        # Curvature feedforward: a second, farther-out lookahead (smaller
        # fraction, since warped y=0 is the horizon) whose lane-center error
        # is blended into steering so the car starts turning as the curve
        # appears instead of only reacting once cross-track error builds up
        # at the near lookahead. Reuses the same error/sign convention as the
        # main PID term rather than a separately-signed curvature constant.
        self.ff_lookahead = float(np.clip(getattr(cfg, "LANE_V2_FF_LOOKAHEAD", 0.30), 0.0, 0.98))
        self.curvature_ff_gain = float(getattr(cfg, "LANE_V2_CURVATURE_FF_GAIN", 0.005))
        # The feedforward extrapolates the boundary fits out to the far lookahead
        # row. On curves that extrapolation is unreliable: the fit swings outside
        # the frame and the apparent lane width balloons, yielding a preview
        # center hundreds of px off that winds steering to full lock. Only trust
        # the preview when the extrapolated width stays within this fraction of
        # the expected width (looser than the near-row tolerance, since a real
        # curve does widen the apparent far width somewhat).
        self.ff_width_tolerance = float(
            getattr(cfg, "LANE_V2_FF_WIDTH_TOLERANCE", 0.25)
        )
        # When a dual pair is rejected, which single boundary to trust. The
        # dashed yellow line is the calibrated-reliable boundary; the white curb
        # is fragile (diagonal, broken, picks up far-field glare) and, being a
        # solid stripe, always has more pixels than the dashes -- so the default
        # "keep the higher-pixel boundary" heuristic keeps white and centers off
        # it, cutting the car right (off the lane) exactly when the yellow inner
        # line leaves the ROI on a left bend. Prefer yellow instead.
        self.single_prefer_yellow = bool(
            getattr(cfg, "LANE_V2_SINGLE_PREFER_YELLOW", True)
        )
        # On tracks where the dashed yellow line is the primary guide, the white
        # curb alone is not a trustworthy lane reference: on a bend the yellow
        # inner line leaves the frame and the follower, left with only the curb,
        # drives the inferred center right off the frame (veers off). When this
        # is set, a frame with no yellow is treated as a detection dropout
        # (bridged by "holding" using the last good fit, then stopped on timeout)
        # instead of trusting a white-only center.
        self.yellow_is_primary = bool(
            getattr(cfg, "LANE_V2_YELLOW_IS_PRIMARY", False)
        )
        # Max the raw lane center may move in a single frame (reference px,
        # scaled to the runtime warp). A crawl-speed car cannot shift the lane
        # center 200 px in one 20 Hz frame -- such jumps are detection glitches
        # (a bad single-boundary fit, a momentary layout flip) and, acted on,
        # they command spurious hard steering that walks the car off-lane. 0
        # disables the clamp (default, so existing behavior/tests are unchanged).
        self.max_center_jump_ref = float(
            getattr(cfg, "LANE_V2_MAX_CENTER_JUMP_PX", 0.0)
        )
        self.max_center_jump = 0.0
        # Curve-memory: deep in a bend both boundaries can thin out of view. The
        # plain recovery path straightens (steering decays) and stops, halting
        # the car mid-turn. Instead, if we were actively turning when the lane
        # was lost, keep applying the last committed turn and crawl through the
        # short blind arc until the lane re-acquires -- bounded by curve_memory_s
        # (measured from the last detection; the recovery_timeout holding window
        # is the first part of it) so a genuine loss still stops. 0 disables it.
        self.curve_memory_s = float(getattr(cfg, "LANE_V2_CURVE_MEMORY_S", 0.0))
        self.curve_memory_min_steer = float(
            getattr(cfg, "LANE_V2_CURVE_MEMORY_MIN_STEER", 0.15)
        )
        self._committed_steering = 0.0
        self.ema_alpha = float(np.clip(getattr(cfg, "LANE_V2_EMA_ALPHA", 0.3), 0.01, 1.0))
        self.recovery_timeout = float(getattr(cfg, "LANE_V2_RECOVERY_TIMEOUT_S", 0.45))

        self.conf_stop = float(getattr(cfg, "LANE_V2_CONFIDENCE_STOP", 0.20))
        self.conf_crawl = float(getattr(cfg, "LANE_V2_CONFIDENCE_CRAWL", 0.50))
        self.conf_normal = float(getattr(cfg, "LANE_V2_CONFIDENCE_NORMAL", 0.80))
        self.allow_single_boundary = bool(
            getattr(cfg, "LANE_V2_ALLOW_SINGLE_BOUNDARY", False)
        )
        self.support_pixels = float(getattr(cfg, "LANE_V2_SUPPORT_PIXELS", 180))
        self.temporal_scale = float(getattr(cfg, "LANE_V2_TEMPORAL_SCALE_PX", 30))

        self.steering_sign = float(getattr(cfg, "LANE_V2_STEERING_SIGN", 1.0))
        self.steering_rate = float(getattr(cfg, "LANE_V2_STEERING_RATE_LIMIT", 0.08))
        self.pid_limit = float(getattr(cfg, "LANE_V2_PID_OUTPUT_LIMIT", 1.0))
        self.pid_st.setpoint = 0.0
        self.pid_st.output_limits = (-self.pid_limit, self.pid_limit)
        self.pid_st.sample_time = None

        self.throttle_normal = float(getattr(cfg, "LANE_V2_THROTTLE_NORMAL", 0.45))
        self.throttle_medium = float(getattr(cfg, "LANE_V2_THROTTLE_MEDIUM", 0.35))
        self.throttle_turn = float(getattr(cfg, "LANE_V2_THROTTLE_TURN", 0.25))
        self.throttle_crawl = float(getattr(cfg, "LANE_V2_THROTTLE_CRAWL", 0.15))
        self.throttle_accel = float(getattr(cfg, "LANE_V2_THROTTLE_ACCEL_LIMIT", 0.03))
        self.throttle_decel = float(getattr(cfg, "LANE_V2_THROTTLE_DECEL_LIMIT", 0.08))
        self.turn_steering = float(getattr(cfg, "LANE_V2_TURN_STEERING", 0.35))
        self.sharp_steering = float(getattr(cfg, "LANE_V2_SHARP_STEERING", 0.65))
        self.curvature_slow = float(getattr(cfg, "LANE_V2_CURVATURE_SLOW", 0.20))

        self.debug_enabled = bool(getattr(cfg, "LANE_FOLLOW_DEBUG", True))
        self.log_every_n = max(1, int(getattr(cfg, "LANE_FOLLOW_LOG_EVERY_N", 5)))
        car_path = getattr(cfg, "CAR_PATH", os.path.expanduser("~/mycar"))
        self.log_path = getattr(
            cfg, "LANE_FOLLOW_LOG_PATH", os.path.join(car_path, "logs", "lane_follow.log")
        )
        self.csv_path = getattr(
            cfg, "LANE_V2_CSV_LOG_PATH", os.path.join(car_path, "logs", "lane_follow_v2.csv")
        )
        self.csv_enabled = bool(getattr(cfg, "LANE_V2_CSV_LOG", True))
        if self.debug_enabled:
            _ensure_file_logger(self.log_path)

        self._warp_key = None
        self._matrix = None
        self._inverse_matrix = None
        self._runtime_shape = None
        self._x_scale = 1.0
        self._y_scale = 1.0
        self._area_scale = 1.0
        self._frame_i = 0
        self._last_time = None
        self._fps = 0.0
        self._last_detection_time = None
        self._last_left_fit = None
        self._last_right_fit = None
        self._lane_layout = None
        self._filtered_center = None
        self._filtered_width = self.expected_lane_width
        self._filtered_error = None
        self._filtered_preview_error = None
        self._previous_raw_center = None
        self.steering = 0.0
        self.throttle = 0.0
        self._dbg = {}

        logger.info(
            "LaneFollower v2 init roi=%.2f lookahead=%.2f expected_width=%.1f "
            "HSV dashed=%s..%s white=%s..%s PID=(%.5f,%.5f,%.5f)",
            self.roi_top, self.lookahead, self.expected_lane_width,
            tuple(self.yellow_lo), tuple(self.yellow_hi),
            tuple(self.white_lo), tuple(self.white_hi),
            getattr(pid, "Kp", 0), getattr(pid, "Ki", 0), getattr(pid, "Kd", 0),
        )

    def apply_hsv_ranges(self, yellow_range=None, white_range=None):
        """Hot-update HSV ranges from the existing live calibration UI."""
        if yellow_range is not None:
            lo, hi = yellow_range
            self.yellow_lo = np.asarray(lo, dtype=np.uint8)
            self.yellow_hi = np.asarray(hi, dtype=np.uint8)
        if white_range is not None:
            lo, hi = white_range
            self.white_lo = np.asarray(lo, dtype=np.uint8)
            self.white_hi = np.asarray(hi, dtype=np.uint8)
        logger.info(
            "LaneFollower v2 HSV updated dashed=%s..%s white=%s..%s",
            tuple(self.yellow_lo), tuple(self.yellow_hi),
            tuple(self.white_lo), tuple(self.white_hi),
        )

    def _perspective(self, width, height):
        key = (width, height, tuple(self.perspective_points.ravel()))
        if key != self._warp_key:
            src = self.perspective_points * np.asarray((width - 1, height - 1), np.float32)
            dst = np.asarray((
                (0, height - 1), (0, 0), (width - 1, 0), (width - 1, height - 1)
            ), dtype=np.float32)
            self._matrix = cv2.getPerspectiveTransform(src, dst)
            self._inverse_matrix = cv2.getPerspectiveTransform(dst, src)
            self._warp_key = key
        return self._matrix

    def _masks(self, warped_rgb):
        hsv = cv2.cvtColor(warped_rgb, self.hsv_code)
        blue = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
        if np.any(self.yellow2_hi):
            blue = cv2.bitwise_or(
                blue, cv2.inRange(hsv, self.yellow2_lo, self.yellow2_hi)
            )
        if self.white_mode == "tophat":
            value = hsv[:, :, 2]
            sat = hsv[:, :, 1]
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.white_tophat_kernel, self.white_tophat_kernel),
            )
            tophat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, k)
            white = (
                (tophat >= self.white_tophat_thresh) & (sat <= self.white_max_sat)
            ).astype(np.uint8) * 255
        else:
            white = cv2.inRange(hsv, self.white_lo, self.white_hi)
        white = cv2.bitwise_and(white, cv2.bitwise_not(blue))
        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        if self.median_kernel >= 3:
            k = _odd(self.median_kernel, 3)
            blue = cv2.medianBlur(blue, k)
            white = cv2.medianBlur(white, k)
        return blue, white

    def _components(self, mask):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        accepted, rejected = [], []
        min_area = self.min_blob_area * self._area_scale
        max_area = self.max_blob_area * self._area_scale
        min_height = self.min_blob_height * self._y_scale
        max_width = self.max_blob_width_frac * mask.shape[1]
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            aspect = float(w) / max(h, 1)
            item = {
                "label": label, "box": (x, y, w, h), "area": area,
                "centroid": tuple(float(v) for v in centroids[label]),
            }
            valid = (
                min_area <= area <= max_area
                and h >= min_height
                and w <= max_width
                and aspect <= self.max_blob_aspect
            )
            (accepted if valid else rejected).append(item)
        return labels, accepted, rejected

    def _ransac_inliers(self, xs, ys, degree):
        """Pick the pixel subset consistent with one thin curve.

        Thin lane markings put nearly every pixel within the residual band of
        a single polynomial, while broad false-positive blobs (walls, glare)
        only contribute a narrow column of inliers, so the marking wins the
        consensus vote even when the noise has more raw pixels.
        """
        threshold = max(self.ransac_threshold * self._x_scale, 1.0)
        n = xs.size
        sample_size = degree + 1
        ys_f = ys.astype(float)
        xs_f = xs.astype(float)
        best = None
        best_count = 0
        for _ in range(self.ransac_iterations):
            idx = self._rng.choice(n, sample_size, replace=False)
            if np.unique(ys_f[idx]).size < sample_size:
                continue
            try:
                coeffs = np.polyfit(ys_f[idx], xs_f[idx], degree)
            except (TypeError, ValueError, np.linalg.LinAlgError):
                continue
            residuals = np.abs(np.polyval(coeffs, ys_f) - xs_f)
            inliers = residuals <= threshold
            count = int(inliers.sum())
            if count > best_count:
                best, best_count = inliers, count
        if best is None or best_count < sample_size + 2:
            return None
        return best

    def _fit_boundary(self, labels, components, image_height):
        if not components:
            return None
        wanted = np.asarray([c["label"] for c in components])
        ys, xs = np.where(np.isin(labels, wanted))
        if xs.size < self.min_fit_pixels * self._area_scale:
            return None
        if xs.size > 3000:
            step = int(np.ceil(xs.size / 3000.0))
            xs, ys = xs[::step], ys[::step]
        unique_y = np.unique(ys).size
        degree = 2 if unique_y >= 6 else 1
        if self.ransac_enabled and xs.size >= (degree + 1) * 3:
            inliers = self._ransac_inliers(xs, ys, degree)
            if inliers is not None:
                xs, ys = xs[inliers], ys[inliers]
                if xs.size < self.min_fit_pixels * self._area_scale:
                    return None
                unique_y = np.unique(ys).size
                degree = 2 if unique_y >= 6 else 1
        y_span = float(np.ptp(ys)) / max(image_height - 1, 1)
        if y_span < self.min_fit_y_span:
            return None
        try:
            coeffs = np.polyfit(ys.astype(float), xs.astype(float), degree)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None
        predicted = np.polyval(coeffs, ys)
        residual = float(np.sqrt(np.mean(np.square(xs - predicted))))
        if (
            not np.isfinite(residual)
            or residual > self.max_fit_residual * self._x_scale
        ):
            return None
        return BoundaryFit(coeffs, degree, residual, int(xs.size), y_span)

    @staticmethod
    def _shift_fit(boundary, delta):
        coeffs = boundary.coeffs.copy()
        coeffs[-1] += delta
        return BoundaryFit(
            coeffs, boundary.degree, boundary.residual, boundary.pixels,
            boundary.y_span, inferred=True,
        )

    def _assign_boundaries(self, yellow, white, width, height):
        """Map color fits to physical left/right boundaries.

        Yellow on the right means the vehicle is in the left lane; yellow on
        the left means it is in the right lane. A previous valid layout is
        preferred during temporary single-color detection.
        """
        look_y = float(np.clip(round(self.lookahead * (height - 1)), 0, height - 1))
        layout = self._lane_layout
        if yellow is not None and white is not None:
            yellow_x = float(yellow.x(look_y))
            white_x = float(white.x(look_y))
            expected = self.expected_lane_width
            if abs(yellow_x - white_x) < 0.15 * expected:
                # Both color masks latched onto the same physical marking
                # (the dashed divider often rides on a white stripe). Keep
                # the dashed fit and let single-boundary inference run.
                white = None
            elif yellow_x < white_x:
                return yellow, white, "right_lane"
            else:
                return white, yellow, "left_lane"

        if layout is None:
            if yellow is not None:
                layout = "left_lane" if yellow.x(look_y) >= width / 2.0 else "right_lane"
            elif white is not None:
                layout = "right_lane" if white.x(look_y) >= width / 2.0 else "left_lane"

        if yellow is not None:
            return (
                (None, yellow, layout)
                if layout == "left_lane"
                else (yellow, None, layout)
            )
        if white is not None:
            return (
                (white, None, layout)
                if layout == "left_lane"
                else (None, white, layout)
            )
        return None, None, layout

    def _geometry(self, left, right, width, height, yellow_side=None):
        # Anchor the validation/inference band to the FIXED calibrated width, not
        # the running filtered width. Using the EMA width here creates a ratchet:
        # a small outward drift of a boundary raises filtered_width, which raises
        # the acceptance band, which admits the next frame's larger drift -- so an
        # inflating lane width is never rejected and single-boundary inference
        # places the missing boundary ever farther out, pulling the center off.
        expected = self.expected_lane_width
        # Validate the pair only over the y-range that actually feeds the
        # controller (the feed-forward preview point through the main
        # lookahead point), not the full ROI. The fits are short polynomials
        # extrapolated well beyond their supporting pixels outside this band;
        # on any real curve that extrapolation swings outside the frame and
        # the horizontal width (measured at fixed y, not perpendicular to the
        # curve) balloons even though the true lane width is constant --
        # rejecting a valid dual detection on every bend, not just bad ones.
        near_y = height * min(self.lookahead, self.ff_lookahead)
        far_y = height * max(self.lookahead, self.ff_lookahead)
        samples = (
            np.linspace(near_y, far_y, 5) if far_y > near_y
            else np.asarray([height * self.lookahead])
        )
        width_score = 0.0
        measured_width = None
        valid_dual = False

        if left is not None and right is not None:
            widths = np.asarray(right.x(samples) - left.x(samples), dtype=float)
            measured_width = float(np.median(widths))
            low = expected * (1.0 - self.width_tolerance)
            high = expected * (1.0 + self.width_tolerance)
            # The controller only ever consumes the lane *midpoint* at the
            # look-ahead rows, so judge the pair on the center line, not on
            # each boundary. On a real curve the outer boundary legitimately
            # runs off the frame edge and the horizontal boundary spacing
            # varies strongly (two parallel curves have constant perpendicular
            # spacing but wildly different horizontal spacing at a fixed y) --
            # gating on either the boundaries' in-frame position or the
            # horizontal width spread rejects nearly every bend. Require the
            # midpoint to stay in-frame and the *median* width to be plausible;
            # crossed/garbage pairs are still caught by widths > 0 and the
            # median-width band.
            centers = (left.x(samples) + right.x(samples)) / 2.0
            center_in_frame = bool(
                np.all(centers >= -0.1 * width) and np.all(centers <= 1.1 * width)
            )
            valid_dual = bool(
                center_in_frame and np.all(widths > 0)
                and low <= measured_width <= high
            )
            if valid_dual:
                width_error = abs(measured_width - expected) / max(expected, 1.0)
                width_score = float(np.clip(1.0 - width_error / self.width_tolerance, 0, 1))
            else:
                # Inconsistent pair (common on sharp curves where the lane
                # width shrinks at the lookahead). Keep one boundary and infer
                # the other instead of dropping both. Prefer the reliable yellow
                # line when we know which side it is; otherwise fall back to the
                # better-supported (higher-pixel) boundary.
                if self.single_prefer_yellow and yellow_side == "left":
                    right = None
                elif self.single_prefer_yellow and yellow_side == "right":
                    left = None
                elif left.pixels >= right.pixels:
                    right = None
                else:
                    left = None
                measured_width = None

        if left is not None and right is None:
            right = self._shift_fit(left, expected)
        elif right is not None and left is None:
            left = self._shift_fit(right, -expected)

        if left is None or right is None:
            return None, None, None, 0.0, valid_dual

        look_y = float(np.clip(round(self.lookahead * (height - 1)), 0, height - 1))
        left_x = float(left.x(look_y))
        right_x = float(right.x(look_y))
        if not (np.isfinite(left_x) and np.isfinite(right_x) and right_x > left_x):
            return None, None, None, 0.0, valid_dual
        center = (left_x + right_x) / 2.0
        if center < -0.1 * width or center > 1.1 * width:
            return None, None, None, 0.0, valid_dual
        lane_width = measured_width if valid_dual else expected
        return left, right, (center, lane_width, look_y), width_score, valid_dual

    def _confidence(self, left_raw, right_raw, width_score, center, held_age=None):
        if held_age is not None:
            base = max(0.0, 1.0 - held_age / max(self.recovery_timeout, 1e-3))
            return min(self.conf_crawl, self.conf_crawl * base)
        detected = int(left_raw is not None) + int(right_raw is not None)
        detection_score = (0.0, 0.62, 1.0)[detected]
        fits = [f for f in (left_raw, right_raw) if f is not None]
        support_score = (
            np.mean([
                min(1.0, f.pixels / max(self.support_pixels * self._area_scale, 1))
                for f in fits
            ])
            if fits else 0.0
        )
        residual_score = (
            np.mean([
                max(
                    0.0,
                    1.0 - f.residual / (self.max_fit_residual * self._x_scale),
                )
                for f in fits
            ])
            if fits else 0.0
        )
        if detected == 1:
            width_score = 0.55
        if self._previous_raw_center is None:
            temporal_score = 1.0
        else:
            temporal_score = max(
                0.0,
                1.0
                - abs(center - self._previous_raw_center)
                / (self.temporal_scale * self._x_scale),
            )
        confidence = (
            0.32 * detection_score + 0.22 * support_score
            + 0.18 * residual_score + 0.16 * width_score
            + 0.12 * temporal_score
        )
        if detected == 1:
            confidence = min(confidence, 0.72)
        return float(np.clip(confidence, 0.0, 1.0))

    def _control(
            self, error, confidence, curvature, lane_valid, movement_allowed,
            cautious=False, preview_error=None):
        pid_p = pid_i = pid_d = 0.0
        if lane_valid and error is not None:
            # Normalize the pixel error to the reference resolution so PID and
            # feedforward gains are resolution-independent: a 480px-wide frame
            # produces ~3x larger pixel errors than the 160px reference, which
            # would otherwise make PID_P ~3x too aggressive and saturate the
            # steering to full lock on ordinary offsets.
            inv_scale = 1.0 / max(self._x_scale, 1e-3)
            raw = self.steering_sign * float(self.pid_st(-float(error) * inv_scale))
            components = getattr(self.pid_st, "components", (raw, 0.0, 0.0))
            pid_p, pid_i, pid_d = (float(v) for v in components)
            if preview_error is not None:
                # Anticipate the curve instead of only reacting to it.
                raw += (self.steering_sign * self.curvature_ff_gain
                        * preview_error * inv_scale)
            raw = float(np.clip(raw, -1.0, 1.0))
            delta = float(np.clip(raw - self.steering, -self.steering_rate, self.steering_rate))
            self.steering = float(np.clip(self.steering + delta, -1.0, 1.0))
        elif confidence <= 0:
            self.steering *= 0.8
            if abs(self.steering) < 0.005:
                self.steering = 0.0

        if not movement_allowed or not lane_valid or confidence < self.conf_stop:
            target = 0.0
        elif confidence < self.conf_crawl:
            target = self.throttle_crawl
        elif confidence < self.conf_normal:
            target = self.throttle_medium
        else:
            target = self.throttle_normal

        if cautious and target > 0:
            target = min(target, self.throttle_crawl)

        steer_abs = abs(self.steering)
        if target > 0 and (steer_abs >= self.sharp_steering or curvature >= self.curvature_slow):
            target = min(target, self.throttle_turn)
        elif target > 0 and steer_abs >= self.turn_steering:
            target = min(target, self.throttle_medium)

        if target == 0:
            self.throttle = 0.0
        else:
            limit = self.throttle_accel if target > self.throttle else self.throttle_decel
            self.throttle += float(np.clip(target - self.throttle, -limit, limit))
        return pid_p, pid_i, pid_d

    def _write_csv(self, row):
        if not self.csv_enabled:
            return
        try:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
            with open(self.csv_path, "a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.CSV_FIELDS)
                if not exists:
                    writer.writeheader()
                writer.writerow({key: row.get(key) for key in self.CSV_FIELDS})
        except OSError as exc:
            logger.warning("Lane v2 CSV write failed: %s", exc)

    def _update_timing(self, now):
        if self._last_time is not None:
            dt = max(now - self._last_time, 1e-6)
            instant = 1.0 / dt
            self._fps = instant if self._fps <= 0 else 0.2 * instant + 0.8 * self._fps
        self._last_time = now

    def _reset_stale_state(self):
        self._last_detection_time = None
        self._last_left_fit = None
        self._last_right_fit = None
        self._lane_layout = None
        self._filtered_center = None
        self._filtered_error = None
        self._filtered_preview_error = None
        self._previous_raw_center = None
        self._committed_steering = 0.0
        reset = getattr(self.pid_st, "reset", None)
        if reset is not None:
            reset()

    def _configure_runtime_scale(self, frame_width, frame_height, roi_width, roi_height):
        shape = (frame_width, frame_height, roi_width, roi_height)
        if shape == self._runtime_shape:
            return
        self._runtime_shape = shape
        self._x_scale = max(roi_width / max(self.reference_width, 1.0), 0.01)
        reference_roi_height = self.reference_height * (1.0 - self.roi_top)
        self._y_scale = max(roi_height / max(reference_roi_height, 1.0), 0.01)
        self._area_scale = self._x_scale * self._y_scale
        self.expected_lane_width = self.expected_lane_width_cfg * self._x_scale
        self._filtered_width = self.expected_lane_width
        self.max_center_jump = self.max_center_jump_ref * self._x_scale
        self._reset_stale_state()
        logger.info(
            "LaneFollower v2 runtime=%dx%d roi=%dx%d scale=(%.2f,%.2f) "
            "expected_width=%.1f",
            frame_width, frame_height, roi_width, roi_height,
            self._x_scale, self._y_scale, self.expected_lane_width,
        )

    def run(self, cam_img):
        self._frame_i += 1
        started = time.monotonic()
        wall_time = time.time()
        self._update_timing(started)
        if cam_img is None:
            self.throttle = 0.0
            self.steering = 0.0
            self._reset_stale_state()
            self._dbg = {"state": "no_image", "latency_ms": 0.0, "fps": self._fps}
            logger.warning("LaneFollower v2 frame=%d no image; stopping", self._frame_i)
            return 0.0, 0.0, None

        frame_h, frame_w = cam_img.shape[:2]
        roi_y = int(round(frame_h * self.roi_top))
        roi = cam_img[roi_y:, :, :]
        roi_h, roi_w = roi.shape[:2]
        self._configure_runtime_scale(frame_w, frame_h, roi_w, roi_h)
        warped = cv2.warpPerspective(
            roi, self._perspective(roi_w, roi_h), (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
        )
        blue_mask, white_mask = self._masks(warped)
        blue_labels, blue_components, blue_rejected = self._components(blue_mask)
        white_labels, white_components, white_rejected = self._components(white_mask)

        yellow_raw = self._fit_boundary(blue_labels, blue_components, roi_h)
        white_raw = self._fit_boundary(white_labels, white_components, roi_h)
        # With yellow as the primary guide, a white-only frame is a dropout, not
        # a usable single-boundary detection (see yellow_is_primary). Drop the
        # white fit so the pipeline bridges with the last good fit and stops on a
        # sustained loss rather than chasing the curb off-lane.
        if self.yellow_is_primary and yellow_raw is None:
            white_raw = None
        left_raw, right_raw, proposed_layout = self._assign_boundaries(
            yellow_raw, white_raw, roi_w, roi_h
        )
        # Which physical side the yellow line is on, so single-boundary inference
        # can prefer it: right_lane layout puts yellow on the left, left_lane on
        # the right.
        yellow_side = (
            "left" if proposed_layout == "right_lane"
            else "right" if proposed_layout == "left_lane"
            else None
        )
        left, right, geometry, width_score, valid_dual = self._geometry(
            left_raw, right_raw, roi_w, roi_h, yellow_side=yellow_side
        )
        state = "lost"
        held_age = None
        if geometry is not None:
            raw_center, measured_width, look_y = geometry
            confidence = self._confidence(
                left_raw, right_raw, width_score, raw_center
            )
            self._last_detection_time = started
            self._last_left_fit, self._last_right_fit = left, right
            self._lane_layout = proposed_layout
            self._previous_raw_center = raw_center
            state = "dual" if valid_dual else "single_inferred"
        elif self._last_detection_time is not None:
            held_age = started - self._last_detection_time
            if held_age <= self.recovery_timeout:
                left, right = self._last_left_fit, self._last_right_fit
                look_y = float(round(self.lookahead * (roi_h - 1)))
                raw_center = float((left.x(look_y) + right.x(look_y)) / 2.0)
                measured_width = self._filtered_width
                confidence = self._confidence(None, None, 0.0, raw_center, held_age)
                state = "holding"
            elif (self.curve_memory_s > self.recovery_timeout
                  and held_age <= self.curve_memory_s
                  and abs(self._committed_steering) >= self.curve_memory_min_steer):
                # Mid-turn loss: coast the committed turn through the blind arc.
                # No detection, so no center/error; steering is pinned to the
                # last committed turn below and throttle held at crawl. Do NOT
                # reset stale state so the lane can re-acquire on curve exit.
                confidence = max(
                    self.conf_stop,
                    0.5 * (self.conf_stop + self.conf_crawl),
                )
                raw_center = measured_width = look_y = None
                left = right = None
                state = "curve_memory"
            else:
                confidence = 0.0
                raw_center = measured_width = look_y = None
                left = right = None
                state = "timeout"
                self._reset_stale_state()
        else:
            confidence = 0.0
            raw_center = measured_width = look_y = None

        if raw_center is not None:
            # Bound a single frame's center move to reject glitch jumps (a bad
            # fit / layout flip snapping the center across the frame) before they
            # reach the EMA and the steering. Only applies once a filtered center
            # exists and the clamp is enabled (max_center_jump > 0).
            if self._filtered_center is not None and self.max_center_jump > 0:
                raw_center = float(np.clip(
                    raw_center,
                    self._filtered_center - self.max_center_jump,
                    self._filtered_center + self.max_center_jump,
                ))
            self._filtered_center = _ema(self._filtered_center, raw_center, self.ema_alpha)
            self._filtered_width = _ema(
                self._filtered_width, measured_width, self.ema_alpha
            )
            raw_error = self._filtered_center - roi_w / 2.0
            self._filtered_error = _ema(self._filtered_error, raw_error, self.ema_alpha)
        else:
            self._filtered_error = None

        preview_error = None
        if left is not None and right is not None:
            far_y = float(np.clip(round(self.ff_lookahead * (roi_h - 1)), 0, roi_h - 1))
            far_left = left.x(far_y)
            far_right = right.x(far_y)
            far_width = abs(far_right - far_left)
            far_center = (far_left + far_right) / 2.0
            # Reject a ballooned/out-of-frame extrapolation before it can wind the
            # feedforward to full lock (see ff_width_tolerance above). Require the
            # extrapolated far width to stay near the expected lane width and the
            # predicted far center to remain within a margin of the frame.
            lo = self.expected_lane_width * (1.0 - self.ff_width_tolerance)
            hi = self.expected_lane_width * (1.0 + self.ff_width_tolerance)
            center_in_frame = -0.1 * roi_w <= far_center <= 1.1 * roi_w
            if lo <= far_width <= hi and center_in_frame:
                raw_preview_error = far_center - roi_w / 2.0
                self._filtered_preview_error = _ema(
                    self._filtered_preview_error, raw_preview_error, self.ema_alpha
                )
                preview_error = self._filtered_preview_error
            else:
                self._filtered_preview_error = None
        else:
            self._filtered_preview_error = None

        curvature = 0.0
        real_fits = [f for f in (left, right) if f is not None and f.degree == 2]
        if real_fits:
            curvature = float(np.mean([
                abs(2.0 * fit.coeffs[0]) * roi_h for fit in real_fits
            ]))
        lane_valid = raw_center is not None and state != "timeout"
        # "holding" bridges a brief detection dropout using the last good fit;
        # it must be allowed to crawl, not stop dead, or the car stalls on
        # every flicker (very common mid-turn). It is inherently time-limited:
        # holding confidence decays with age and the state becomes "timeout"
        # (which stops) once the dropout exceeds the recovery window.
        cautious = state in ("single_inferred", "holding", "curve_memory")
        if state == "curve_memory":
            # Pin steering to the committed turn (no detection to steer from) and
            # allow crawl so the car coasts the arc instead of straightening.
            self.steering = self._committed_steering
            movement_allowed = True
        else:
            movement_allowed = valid_dual or (
                self.allow_single_boundary and cautious
            )
        pid_p, pid_i, pid_d = self._control(
            self._filtered_error, confidence, curvature, lane_valid,
            movement_allowed, cautious, preview_error=preview_error,
        )
        # Remember the live turn while genuinely tracking, so curve-memory has a
        # direction to hold if the lane is lost mid-bend.
        if state in ("dual", "single_inferred"):
            self._committed_steering = self.steering
        latency_ms = (time.monotonic() - started) * 1000.0

        self._dbg = {
            "roi_y": roi_y, "warped": warped, "blue_mask": blue_mask,
            "white_mask": white_mask, "blue_components": blue_components,
            "white_components": white_components, "blue_rejected": blue_rejected,
            "white_rejected": white_rejected, "left": left, "right": right,
            "look_y": look_y, "lane_center": self._filtered_center if lane_valid else None,
            "image_center": roi_w / 2.0, "error": self._filtered_error,
            "lane_width": self._filtered_width, "confidence": confidence,
            "curvature": curvature, "pid": (pid_p, pid_i, pid_d),
            "state": state, "lane_layout": self._lane_layout,
            "latency_ms": latency_ms, "fps": self._fps,
            "roi_shape": (roi_h, roi_w),
        }

        row = {
            "timestamp": f"{wall_time:.6f}", "frame": self._frame_i,
            "lane_center": self._dbg["lane_center"],
            "image_center": roi_w / 2.0, "offset": self._filtered_error,
            "confidence": confidence, "lane_width": self._filtered_width,
            "left_detected": int(left_raw is not None),
            "right_detected": int(right_raw is not None),
            "pid_p": pid_p, "pid_i": pid_i, "pid_d": pid_d,
            "steering": self.steering, "throttle": self.throttle,
            "fps": self._fps, "latency_ms": latency_ms, "state": state,
        }
        self._write_csv(row)

        if self.debug_enabled and (
            self._frame_i % self.log_every_n == 0 or state in ("lost", "timeout")
        ):
            # held = which real boundary survives (the other is inferred/None),
            # so we can see whether a right-drift is the follower leaning on the
            # white curb (held=white) vs the yellow line itself moving.
            if left is not None and right is not None:
                real_side = ("L" if not getattr(left, "inferred", False) else "") \
                    + ("R" if not getattr(right, "inferred", False) else "")
            else:
                real_side = "L" if left is not None else "R" if right is not None else "-"
                real_side += "?"
            held_color = "?"
            if real_side and real_side[0] in ("L", "R") and yellow_side is not None:
                held_side = real_side[0]
                held_color = "yellow" if (
                    (held_side == "L" and yellow_side == "left")
                    or (held_side == "R" and yellow_side == "right")
                ) else "white"
            logger.info(
                "V2 frame=%d state=%s center=%s err=%s width=%.1f conf=%.2f "
                "steer=%.3f throttle=%.3f y=%d w=%d held=%s(%s) yside=%s "
                "fps=%.1f latency=%.1fms",
                self._frame_i, state,
                None if self._dbg["lane_center"] is None else round(self._dbg["lane_center"], 1),
                None if self._filtered_error is None else round(self._filtered_error, 1),
                self._filtered_width, confidence, self.steering, self.throttle,
                int(yellow_raw is not None), int(white_raw is not None),
                real_side, held_color, yellow_side,
                self._fps, latency_ms,
            )

        output = self.overlay_display(cam_img) if self.overlay_image else cam_img
        return self.steering, self.throttle, output

    @staticmethod
    def _draw_components(image, components, color, thickness):
        for component in components:
            x, y, w, h = component["box"]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    def overlay_display(self, cam_img):
        image = cam_img.copy()
        d = self._dbg
        warped = d.get("warped")
        if warped is None:
            return image
        debug = warped.copy()

        blue = d["blue_mask"] > 0
        white = d["white_mask"] > 0
        debug[blue] = (
            0.35 * debug[blue] + 0.65 * np.asarray((30, 120, 255))
        ).astype(np.uint8)
        debug[white] = (
            0.35 * debug[white] + 0.65 * np.asarray((255, 255, 255))
        ).astype(np.uint8)
        self._draw_components(debug, d["blue_rejected"], (255, 0, 0), 1)
        self._draw_components(debug, d["white_rejected"], (255, 0, 0), 1)
        self._draw_components(debug, d["blue_components"], (0, 255, 255), 1)
        self._draw_components(debug, d["white_components"], (0, 255, 0), 1)

        roi_h, roi_w = d["roi_shape"]
        ys = np.arange(0, roi_h, dtype=np.float32)
        for fit, color in ((d.get("left"), (0, 180, 255)), (d.get("right"), (255, 255, 255))):
            if fit is None:
                continue
            xs = np.clip(fit.x(ys), 0, roi_w - 1).astype(np.int32)
            points = np.column_stack((xs, ys.astype(np.int32))).reshape((-1, 1, 2))
            cv2.polylines(debug, [points], False, color, 2, cv2.LINE_AA)

        look_y = d.get("look_y")
        lane_center = d.get("lane_center")
        if look_y is not None:
            cv2.line(debug, (0, int(look_y)), (roi_w - 1, int(look_y)), (0, 255, 0), 1)
        if lane_center is not None and look_y is not None:
            cv2.circle(debug, (int(lane_center), int(look_y)), 4, (0, 255, 0), -1)
        cv2.line(
            debug, (int(d["image_center"]), 0), (int(d["image_center"]), roi_h - 1),
            (255, 255, 255), 1,
        )

        # Put the bird's-eye diagnostic in the lower ROI and small mask insets above.
        roi_y = d["roi_y"]
        image[roi_y:, :, :] = cv2.addWeighted(image[roi_y:, :, :], 0.25, debug, 0.75, 0)
        inset_w = max(40, image.shape[1] // 4)
        inset_h = max(25, int(roi_h * inset_w / max(roi_w, 1)))
        blue_rgb = cv2.cvtColor(d["blue_mask"], cv2.COLOR_GRAY2RGB)
        white_rgb = cv2.cvtColor(d["white_mask"], cv2.COLOR_GRAY2RGB)
        image[0:inset_h, 0:inset_w] = cv2.resize(blue_rgb, (inset_w, inset_h))
        image[0:inset_h, inset_w:2 * inset_w] = cv2.resize(
            white_rgb, (inset_w, inset_h)
        )

        p, i, derivative = d["pid"]
        lines = (
            f"V2 {d['state']} {d.get('lane_layout') or 'unknown'} "
            f"conf:{d['confidence']:.2f} fps:{d['fps']:.1f}",
            f"off:{d['error'] if d['error'] is None else round(d['error'], 1)} "
            f"width:{d['lane_width']:.1f} curve:{d['curvature']:.3f}",
            f"PID {p:+.3f} {i:+.3f} {derivative:+.3f}",
            f"steer:{self.steering:+.2f} throttle:{self.throttle:.2f} "
            f"{d['latency_ms']:.1f}ms",
        )
        y = inset_h + 12
        for text in lines:
            cv2.putText(image, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 12
        return image
