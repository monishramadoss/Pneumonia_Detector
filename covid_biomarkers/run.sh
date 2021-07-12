#!/bin/bash
export JARVIS_PATH_CONFIGS=/home/mramados/.jarvis

python /home/mramados/Pneumonia_Detector/covid_biomarkers/dual_loss/model.py > /home/mramados/Pneumonia_Detector/covid_biomarkers/dual_loss/stdout 2>&1
python /home/mramados/Pneumonia_Detector/covid_biomarkers/single_loss/model.py > /home/mramados/Pneumonia_Detector/covid_biomarkers/single_loss/stdout 2>&1
