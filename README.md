# cv-radar

This is a set of computer vision systems for detecting car speeds from video footage, currently from the BrnoCompSpeed dataset.

The first model, in the speed.ipynb file, uses YOLOv5 as its detection model, followed by deepSORT for tracking the cars.

The second model, in the v8track.py file, uses YOLOv8 with native bytetrack tracking.

The third model, in the strongsort_speed.py file, uses YOLOv8 for detection and StrongSORT for tracking.
## Requirements

Python, Jupyter Kernel, opencv-python, torch, ultralytics deep_sort_realtime, strongsort-pip 

## Dataset

Learn about the BrnoCompSpeed dataset from this paper: https://arxiv.org/pdf/1702.06441.pdf
