# Development Notes

## Single-line follower

The controller searches one or more horizontal bands for a colored tape component. Candidates are filtered by size, shape, color dominance, confidence, and distance from the previous target. A PID controller turns horizontal error into steering. If no valid component is accepted, steering and throttle return to zero.

Important configuration groups:

- camera input order and reference resolution
- HSV threshold ranges
- scan-band position and height
- component size and quality limits
- tracking, reacquisition, and smoothing
- PID gains and throttle limits

## Lane follower v2

The controller crops the image, applies a perspective transform, and builds separate masks for the two lane-boundary colors. It fits boundary curves, estimates a lane center, checks the inferred lane width, and computes a confidence score. Confidence and curvature affect throttle, while PID and rate limiting control steering.

The controller can temporarily estimate the missing side from the detected boundary and recent lane width. That recovery is intentionally time-limited.

## What I would improve next

1. Record every run with synchronized raw frames, controller output, and configuration values.
2. Use a fixed test matrix for lighting, speed, and camera angle.
3. Compare revisions with the same starting position and conditions.
4. Report lap-completion rate over multiple trials, not only the best run.
5. Add failure categories such as lost boundary, false detection, steering saturation, and insufficient throttle.

Those measurements would make future debugging claims easier to support and would show whether a change improves reliability instead of only producing one successful run.
