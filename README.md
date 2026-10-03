# DonkeyCar Vision Control

This is a smaller, readable version of the vision-control work I tested on our UC San Diego DonkeyCar. It focuses on two behaviors: following one taped line and driving between two lane boundaries.

The full class repository contains the complete DonkeyCar framework, deployment scripts, AprilTag work, and experiments. This repo keeps only the two controllers, their offline tests, and the notes needed to understand my work.

## Results

| Behavior | Best physical-track result | Honest limitation |
| --- | --- | --- |
| Single-line following | Completed three consecutive laps | Tuned for our car, camera position, tape color, and track conditions |
| Lane following | Completed one full lap | Did not complete three consecutive laps |

These are course-project results, not formal benchmarks. I will add the original driving clips below when I finish organizing them.

> ### Single-line following — three consecutive laps

[▶ Watch driving video](https://drive.google.com/file/d/1GkL3dsHEKlm3UNOgTWRxHftuGGPT9M7J/view?usp=sharing)

> ### Lane following — one full lap

[▶ Watch driving video](https://drive.google.com/file/d/1LPARtrPd4J6LUpSnXVZjDra-EsLTS1lM/view?usp=sharing)

## My role

This was a two-person team project. My clearest individual contribution was the single-line follower. I also participated in physical testing and iteration for the lane follower. I ran the car, checked the camera and debug output, compared revisions on the track, and kept the versions that performed best in our available track sessions.

I also used GPT/Codex extensively. The coding agent proposed and wrote many of the detailed diagnoses and code changes. I was responsible for running the hardware tests and deciding whether a revision helped. Because we did not isolate every variable in a controlled experiment, I do not claim that every suggested cause was proven.

## System

- DonkeyCar-based RC platform
- Raspberry Pi
- OAK-D camera
- VESC motor controller and steering servo
- Python, OpenCV, NumPy, and PID control

Both controllers use the same basic loop:

1. Read a camera frame.
2. Segment track markings with HSV color thresholds.
3. Estimate a target position relative to the image center.
4. Convert the position error into steering with PID control.
5. Adjust or stop throttle based on detection confidence.

The lane follower adds a bird's-eye transform, separate boundary masks, curve fitting, lane-width checks, and short-term recovery when only one boundary is visible.

## What I learned

The most useful part of this project was seeing the gap between code that looks reasonable and a robot that can finish a lap. Camera color order, lighting, image resolution, track geometry, controller state, motor response, and speed all interact on the physical car.

The code includes safeguards for several plausible failure modes:

- BGR/RGB handling before HSV thresholding
- rejection of blobs that do not look like tape or lane boundaries
- scaling of pixel-based settings when the camera resolution changes
- zero steering and throttle when no valid target is accepted
- confidence-based speed reduction for uncertain lane estimates
- limited recovery from a missing boundary using recent lane geometry

These are real engineering concerns, but this repository does not prove that each one caused a particular failed run. That distinction matters to me: an observed driving result is evidence; a debugging explanation is still a hypothesis until it is tested directly.

## Repository layout

```text
line_following/
  line_follower.py              # single taped-line controller
  test_line_follower.py         # focused synthetic-image tests
lane_following/
  lane_follower_v2.py           # two-boundary lane controller
  test_lane_follower_v2.py      # synthetic geometry and recovery tests
docs/
  EVIDENCE.md                   # verified results vs. open claims
  DEVELOPMENT_NOTES.md          # concise technical notes
media/
  README.md                     # where to add driving videos
```

## Run the offline tests

The tests use generated images. They check perception and control behavior without requiring the car.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

These tests are useful regression checks, but they are not a substitute for driving the physical car.

## Original project

- [Team 2 final project repository](https://github.com/UCSD-Silberman-Classes-and-Projects/UCSD-DSC190-SUMMER_I-Final_Project-Team_02)
- [DonkeyCar](https://github.com/autorope/donkeycar)

See [ATTRIBUTION.md](ATTRIBUTION.md) for project scope and code provenance.
