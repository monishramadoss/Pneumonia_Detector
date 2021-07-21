#!/bin/bash
export JARVIS_PATH_CONFIGS=/home/mramados/.jarvis
python /home/mramados/Pneumonia_Detector/covid_biomarkers/vit_model.py > /home/mramados/Pneumonia_Detector/covid_biomarkers/vit_loss_stdout 2>&1
