#!/bin/bash

#block(name=experiment11, threads=5, memory=20000, subtasks=1, gpu=true, hours=100)
   python -u train_FlowNet_ms_warping.py  --dataset_root ./dataset/vimeo_septuplet/ --dataset_crop False
   echo "Done" 

# if you want to schedule multiple gpu jobs on a server, better to use this tool.
# run: `bash ./qsub-SurfaceNet_inference.sh`
# for installation & usage, please refer to the author's github: https://github.com/alexanderrichard/queueing-tool
    