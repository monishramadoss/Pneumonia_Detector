#!/bin/bash
export JARVIS_PATH_CONFIGS=/home/mramados/.jarvis

python /home/mramados/Pneumonia_Detector/dual_loss/model.py > /home/mramados/Pneumonia_Detector/dual_loss/stdout 2>&1
python /home/mramados/Pneumonia_Detector/single_loss/model.py > /home/mramados/Pneumonia_Detector/single_loss/stdout 2>&1
