## Few-Shot Medical Image Segmentation via Dual-Stream Feature  Extractor and Detail-Enhanced Prototype Transformer


**Abstract**:

Few-shot segmentation (FSS) has emerged as a promising approach for medical image analysis, especially in scenarios where annotated data is limited. However, the performance of existing prototypical networks is typically hindered by two critical challenges: 1) intra-class variation arising from feature distribution discrepancies between support and query sets. 2) Extreme inter-class imbalance, often exacerbated by complex background structures. To tackle these challenges, we propose an innovative dual-stream prototype optimization framework, the Dual-stream Feature Extractor and Detail-enhancing prototype Transformer (DFDT). Firstly, we introduce a Dual-stream Visual Feature Fusion (DVFF) module that simultaneously merges local and global features, generating detailed, semantically rich interactions. This module enables the initial prototype to capture global semantics while preserving fine-grained details, effectively reducing intra-class variation. Subsequently, the iterative Prototype Detail Enhancement Transformer (PDET) is designed, employing iterative cycles of bias correction and detail enhancement. This component systematically filters confounding background noise from the prototype while reinforcing core information pertinent to the target foreground, significantly improving robustness against variations in complex backgrounds. Extensive experiments on three public medical imaging datasets demonstrate that the proposed DFDT model achieves significant performance gains over current state-of-the-art methods.
**NOTE: We are actively updating this repository**

If you find this code base useful, please cite our paper. Thanks!


### 1. Dependencies

Please install essential dependencies (see `requirements.txt`) 

```
dcm2nii
nibabel==2.5.1
numpy==1.21.6
opencv-python==4.1.1
Pillow==9.5.0 
sacred==0.7.5
scikit-image==0.14.0
SimpleITK==1.2.3
torch==1.8.1
torchvision==0.9.1
```

### 2. Data pre-processing 

### Datasets and pre-processing
**NOTE:** The ipynb and sh files below, used for pre-processing, can be found at the link of SSL-ALPNet: https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation 

Download:  
**Abdominal MRI**

0. Download [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/) and put the `/MR` folder under `./data/CHAOST2/` directory

1. Converting downloaded data (T2 fold) to `nii` files in 3D for the ease of reading

run `./data/CHAOST2/dcm_img_to_nii.sh` to convert dicom images to nifti files.

run `./data/CHAOST2/png_gth_to_nii.ipynp` to convert ground truth with `png` format to nifti.

2. Pre-processing downloaded images

run `./data/CHAOST2/image_normalize.ipynb`

**Abdominal CT**

0. Download [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and put the `/img` and `/label` folders under `./data/SABS/` directory

1. Intensity windowing 

run `./data/SABS/intensity_normalization.ipynb` to apply abdominal window.

2. Crop irrelavent emptry background and resample images

run `./data/SABS/resampling_and_roi.ipynb` 

**Shared steps**

3. Build class-slice indexing for setting up experiments

run `./data/<CHAOST2/SABS>class_slice_index_gen.ipynb`  

### Training  
1. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints `./pretrained_model/hub/checkpoints` folder,
2. Run `training.py` 
 

### Testing
Run `train_SABS.py`
    `train_CHAOST2.py`
    `train_CMR.py`

### Acknowledgement
This code is based on [SSL-ALPNet](https://arxiv.org/abs/2007.09886v2) (ECCV'20) by [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git)
