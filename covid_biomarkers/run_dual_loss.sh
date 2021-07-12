#!/bin/bash
export JARVIS_PATH_CONFIGS=/home/mramados/.jarvis
python /home/mramados/Pneumonia_Detector/covid_biomarkers/dual_loss.py > /home/mramados/Pneumonia_Detector/covid_biomarkers/dual_loss_stdout 2>&1
