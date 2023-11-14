# TinyPower

## Reference
When reporting results that use the dataset or code in this repository, please cite the paper below:

Haipeng Li, Mabon Ninan, Boyang Wang, John M. Emmert, "TinyPower: Side-Channel Attacks with Tiny Neural Networks," IEEE International Symposium on Hardware Oriented Security and Trust (HOST 2024), Washington DC, USA, May 6-9 2024

## Requirements
This project is written in Python 3.8 and Tensorflow 2.3.1 . Our experiments is running with:

* GPU machine (Intel i9 CPU, 64GB memory,and a NVIDIA Titan RTX GPU).
* Raspberry Pi 4 Model B running Debian Version 11 (64-bit) with Broadcom BCM2711, quad-core Cortex-A72 64-bit 1.8GHz, and 4GB SDRAM.
  
## Reproduce Our Results
### Code 
The codebased include 2 folders: MiniDrop, and Unstructured_Pruning SCA
  * MiniDrop
    * Score_Algo: Code to generate CNN filter scores using l-2 or FPGM  (More info given in MiniDrop/Score_Algo/README.md)
    * cnn: Code for Structured, automatic and baseline cnn train and test (More info given in MiniDrop/README.md)
    * model_performance: This fodler contains various scrips that can be used to measure CNN Model performance  (More info given in MiniDrop/model_performance/README.md)
    * utils: aditional tools for code   
  * Unstructured_Pruning SCA: CNN implimentation for Unstructured Pruning (More info given in Unstructured_Pruning/README.md)
    * cnn
    * utils: aditional tools for code


### Datasets
Our datasets used in this study can be accessed through the link below (last modified Nov 2023):

TBD 

Note: the above link need to be updated every 6 months due to certain settings of OneDrive. If you find the links are expired and you cannot access the data, please feel free to email us (boyang.wang@uc.edu). We will be update the links as soon as we can. Thanks!


### How to Reproduce the results
1.  For CNN based method, please follow the description in cnn/README.md

# Contacts
Mabon Ninan ninanmm@mail.uc.edu

Boyang Wang wang2ba@ucmail.uc.edu

Haipeng Li li2hp@mail.uc.edu
