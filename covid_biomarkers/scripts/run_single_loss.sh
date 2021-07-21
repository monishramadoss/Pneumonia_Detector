#!/bin/bash
export JARVIS_PATH_CONFIGS=/home/mramados/.jarvis
python /home/mramados/Pneumonia_Detector/covid_biomarkers/single_loss.py > /home/mramados/Pneumonia_Detector/covid_biomarkers/single_loss_stdout.log 2>&1
