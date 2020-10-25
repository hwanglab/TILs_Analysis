# TILs_Analysis
## This repository incldues all the codes I developed for TILs detection and related analysis. 

### Notes
(1) If you only want to use my trained TILs detector to detect tils regions in the WSI, you can directly run and learn examples provided in the link: https://github.com/hwanglab/wsi_deploy_models

(2) If you want to check the process how to train tils detecors, it would involve multiple steps: 
    
Step-1: I downloaded and used the training dataset from the paper: "Spatial Organization and Molecular Correlation of Tumor-Infiltrating Lymphocytes Using Deep Learning on Pathology Images". You probably need to read and understand the dataset from this paper.

Step-2: You can run the training and testing procoess from the file: main_tils_train_test.py. In the training process, I used the dataset built by myself (train, valid and test), which could be found in the lab share space: Z:\Datasets\Pathology_Slides\pan_cancer_tils. For testing example, you can refer it from https://github.com/hwanglab/wsi_deploy_models

### Others
(1) Invasive Margin Analysis: We worked with Dr.Kang for quantifying tils density at tumor invasive margins. For codes I wrote, you can find them from the file: main_tils_analysis_v01.py

(2) KM analysis for tils density at invasive margins: For the R codes I worte for KM analysis of colon cancer patient survivals (collaborated with Dr.Kang), you could find them at the locaton: ./R_analysis

(3) Entropy Computation: Given the heatmap predictions, we can compute the entropy to quantify its heterogeneities. The example can be found the in the location: ./Utility_debugs/com_entropy.py. The function I wrote to compute Shannon entropy is: def shannon_entropy_bin(X,b=0.1,vmin=0.0,vmax=1.0):

