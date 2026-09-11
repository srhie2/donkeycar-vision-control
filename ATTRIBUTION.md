# Attribution and Project Scope

This repository is a curated portfolio extract from the UC San Diego DSC 190 Summer Session I Team 2 final project:

- Original team repository: https://github.com/UCSD-Silberman-Classes-and-Projects/UCSD-DSC190-SUMMER_I-Final_Project-Team_02
- Team members: Aung and Sungsan Rhie
- Portfolio repository maintainer: Sungsan Rhie

The original project was built on DonkeyCar. Its MIT license and copyright notice are preserved in `LICENSE`.

## Files in this extract

- `line_following/line_follower.py` was copied from the final line-following submission preserved in the course workspace. The portfolio copy only rewrites one comment so that a debugging hypothesis is not stated as a proven cause. Runtime behavior is unchanged.
- `lane_following/lane_follower_v2.py` was copied from `lane_follower_part.py` in the team repository at commit `fc60f2cdb8a0baad8642e63f29b3ca0dfec686ff`.
- `lane_following/test_lane_follower_v2.py` comes from the same team repository. Its import path was updated for this smaller layout.
- `line_following/test_line_follower.py` contains focused portfolio regression tests adapted to the extracted controller. These tests are not presented as physical-track evidence.

The full deployment configuration and DonkeyCar runtime remain in the original team repository. This extract is meant for code review, not as a drop-in vehicle image.

## AI-assisted development

GPT/Codex was used heavily during the course project and during this portfolio cleanup. It proposed or wrote many code changes and debugging explanations. Physical testing, result selection, and the decision to keep or reject revisions were performed by the student team.
