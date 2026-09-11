# Evidence and Limits

This page separates outcomes that were observed on the car from explanations that remain hypotheses.

## Confirmed during physical testing

- The single-line follower completed three consecutive laps on the course setup used by the team.
- The lane follower completed one full lap on the tested course.
- The lane follower did not complete three consecutive laps.
- Both behaviors ran on the DonkeyCar hardware with an OAK-D camera and VESC drivetrain.

## Verified by offline tests

- The single-line controller detects a centered synthetic yellow marking, rejects a distant initial candidate, stops on an invalid observation, and converts the debug image from BGR to RGB.
- The lane controller's synthetic tests cover dual and single boundaries, curves, glare-like blobs, width checks, temporal recovery, confidence behavior, runtime scaling, and logging fields.

Offline tests are regression checks for the code. They do not prove track reliability.

## Plausible but not isolated as root causes

- color-order mismatch before HSV conversion
- lighting or background objects producing false-positive masks
- pixel thresholds behaving differently at another camera resolution
- old controller state affecting a later frame
- commanded throttle being below the drivetrain's useful response range

These can happen in camera-based mobile robots, and the code contains defenses for them. During this project, however, we did not collect enough controlled before-and-after data to say that every item was the cause of a specific failure.

## Evidence still to add

- **UPLOAD HERE:** original single-line run video
- **UPLOAD HERE:** original one-lap lane-following video
- optional annotated camera frames showing accepted and rejected detections
- a short test table with lighting, speed, camera position, and lap outcome for future runs
