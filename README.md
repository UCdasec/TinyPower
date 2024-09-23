# TinyPower
**The dataset and code are for research purpose only** 

Side-channel attacks leverage correlations between power consumption and intermediate results of encryption to infer encryption keys. Recent studies show that deep learning offers promising results in the context of side-channel attacks. However, neural networks utilized in deep-learning side-channel attacks are complex with substantial amounts of parameters and consume significant memory. As a result, it is challenging to perform deep-learning side-channel attacks on resource-constrained devices. 

This project proposes a framework, named TinyPower, which leverages pruning to reduce the number of parameters of neural networks for side-channel attacks. Pruned neural networks obtained from our framework can successfully run side-channel attacks with a much lower number of parameters and less memory. Specifically, we focus on structured pruning over filters of Convolutional Neural Networks (CNNs). We demonstrate the effectiveness of structured pruning over power and EM traces of AES-128 running on microcontrollers (AVR XMEGA and ARM STM32) and FPGAs (Xilinx Artix-7). Our experimental results show that we can achieve a reduction rate of 98.8\% (e.g., reducing the number of parameters from 53.1 million to 0.59 million) on a CNN and still recover keys on XMEGA. For STM32 and Artix-7, we achieve a reduction rate of 92.9\% and 87.3\% on a CNN respectively. We also demonstrate that our pruned CNNs can effectively perform the attack phase of side-channel attacks on a Raspberry Pi 4 with less than 2.5 millisecond inference time per trace and less than 41 MB memory usage per CNN.  

## Reference
When reporting results that use the dataset or code in this repository, please cite the paper below:

Haipeng Li, Mabon Ninan, Boyang Wang, John M. Emmert, "TinyPower: Side-Channel Attacks with Tiny Neural Networks," IEEE International Symposium on Hardware Oriented Security and Trust (**IEEE HOST 2024**), Washington DC, USA, May 6-9 2024

## Requirements
This project is written in Python 3.8 and Tensorflow 2.3.1 . Our experiments is running with:

* GPU machine (Intel i9 CPU, 64GB memory,and a NVIDIA Titan RTX GPU).
* Raspberry Pi 4 Model B running Debian Version 11 (64-bit) with Broadcom BCM2711, quad-core Cortex-A72 64-bit 1.8GHz, and 4GB SDRAM.
  
## Code and Datasets
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
Our datasets used in this study can be accessed through the link below (**last modified Sept 2024**):

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EoE4ELOgXD1Em7YzFTAQ0ywB_jB1Ic53Qcug9WzZZYj2UA?e=KWvn86

Note: the above link need to be updated every 6 months due to certain settings of OneDrive. If you find the links are expired and you cannot access the data, please feel free to email us (Dr. Boyang Wang, boyang.wang@uc.edu). We will be update the links as soon as we can (typically within 2 days). Thanks!


### How to Reproduce the results
1.  For CNN based method, please follow the description in cnn/README.md

# Contacts
Mabon Ninan ninanmm@mail.uc.edu

Boyang Wang boyang.wang@uc.edu

Haipeng Li li2hp@mail.uc.edu

Research Group Webpage: https://homepages.uc.edu/~wang2ba/
