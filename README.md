# [ECCV 2026] From Local Windows to Adaptive Candidates via Individualized Exploratory: Rethinking Attention for Image Super-Resolution

This repository is an official implementation of the paper "From Local Windows to Adaptive Candidates via Individualized Exploratory: Rethinking Attention for Image Super-Resolution", ECCV, 2026.

[![arXiv](https://img.shields.io/badge/Arxiv-2601.03824-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2601.08341)
[![Pretrained Models](https://img.shields.io/badge/Pretrained%20Models-AD1C18.svg?logo=Googledrive)](https://drive.google.com/drive/folders/1YzKzkGnpvbbzxarnM1fEPXu0z7ojJQrc?usp=sharing)
[![Visual Results](https://img.shields.io/badge/Visual%20Results-AD1C18.svg?logo=Googledrive)](https://drive.google.com/drive/folders/1qS8iY-sBZbikDOnDaWNFQehnB99D67rV?usp=sharing)

By [Chunyu Meng](https://openreview.net/profile?id=%7EChunyu_Meng5), [Wei Long](https://scholar.google.com/citations?user=CsVTBJoAAAAJ) and [Shuhang Gu](https://scholar.google.com/citations?user=-kSTt40AAAAJ).

> **Abstract:** Single Image Super-Resolution (SISR) is a fundamental computer vision task that aims to reconstruct a high-resolution (HR) image from a low-resolution (LR) input. Transformer-based methods have achieved remarkable performance by modeling long-range dependencies in degraded images. However, their feature-intensive attention computation incurs high computational cost. To improve efficiency, most existing approaches partition images into fixed groups and restrict attention within each group. Such group-wise attention overlooks the inherent asymmetry in token similarities, thereby failing to enable flexible and token-adaptive attention computation. To address this limitation, we propose the Individualized Exploratory Transformer (IET), which introduces a novel Individualized Exploratory Attention (IEA) mechanism that allows each token to adaptively select its own content-aware and independent attention candidates. This token-adaptive and asymmetric design enables more precise information aggregation while maintaining computational efficiency. Extensive experiments on standard SR benchmarks demonstrate that IET achieves state-of-the-art performance under comparable computational complexity.
> <img width="980" src="figures/comparison.png">
> 
> <br>
>
> <img width="980" src="figures/attention.png"> 

## 📑 Contents
1. [Environment](#environment)
1. [Installation](#installation)
1. [Inference](#inference)
1. [Training](#training)
1. [Testing](#testing)
1. [Results](#results)
1. [Visual Results](#visual-results)
1. [Visualization of Attention Distributions](#visualization-of-attention-distributions)
1. [Acknowledgements](#acknowledgements)
1. [Citation](#citation)

## <a name="environment"></a> 📢 Environment
- Python 3.11
- PyTorch 2.5.0

## <a name="installation"></a> ⚙️ Installation
1. **Create conda environment and install dependencies:**
   ```bash
   git clone https://github.com/CVL-UESTC/IET.git
  
   conda create -n IET python=3.11
   conda activate IET
  
   pip install -r requirements.txt
   python setup.py develop
   ```

2. **Install the CUDA operators for sparse matrix operations:**
   ```bash
   cd ./ops_smm
   ./make.sh
   cd ..
   ```

3. **Install [Neighborhood Attention](https://natten.org/install/):**

   Make sure to install the older version 0.17.5 or lower, as the new version no longer supports obtaining the attention map.
   You can also use the .whl file provided [here](https://drive.google.com/drive/folders/1jAy1gQ31SdhwkU34b3VcUqQB7QH1hE8i?usp=sharing)
  
   ```bash
   pip install natten-0.17.5+torch250cu121-cp311-cp311-linux_x86_64.whl
   ```

## <a name="inference"></a> 🔧 Inference
Using ```inference.py``` for fast inference on single image or multiple images within the same folder.
```bash
# For classical SR
python inference.py -i inference_image.png -o results/test/ --scale 4 --task classical
python inference.py -i inference_images/ -o results/test/ --scale 4 --task classical

# For lightweight SR
python inference.py -i inference_image.png -o results/test/ --scale 4 --task lightweight
python inference.py -i inference_images/ -o results/test/ --scale 4 --task lightweight
```
The IET SR model processes the image ```inference_image.png``` or images within the ```inference_images/``` directory. The results will be saved in the ```results/test/``` directory.


## <a name="training"></a> 🔩 Training
### Data Preparation
- Download the training dataset DF2K ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)) and put them in the folder `./datasets`.
- It's recommanded to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.

### Training Commands
- Refer to the training configuration files in `./options/train` folder for detailed settings.
- IET (Classical Image Super-Resolution)
```bash
# training dataset: DF2K

# ×2 scratch, input size = 64×64, 500k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/000_IET_SRx2_scratch.yml --launcher pytorch

# ×3 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/002_IET_SRx3_finetune.yml --launcher pytorch

# ×4 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/003_IET_SRx4_finetune.yml --launcher pytorch
```

- IET-light (Lightweight Image Super-Resolution)
```bash
# training dataset: DIV2K

# ×2 scratch, input size = 64×64, 500k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/101_IET_light_SRx2_scratch.yml --launcher pytorch

# ×3 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/102_IET_light_SRx3_finetune.yml --launcher pytorch

# ×4 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/103_IET_light_SRx4_finetune.yml --launcher pytorch
```


## <a name="testing"></a> 🔨 Testing
### Data Preparation
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1_4Fy9emAcqdiBwVM6FvbJU50LCtaBoMt/view?usp=sharing)]) and put them in the folder `./datasets`.

### Pretrained Models
- Download the [pretrained models](https://drive.google.com/drive/folders/1YzKzkGnpvbbzxarnM1fEPXu0z7ojJQrc?usp=sharing) and put them in the folder `./experiments/pretrained_models`.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.
- IET (Classical Image Super-Resolution)
- **We have now integrated the patchwise_testing strategy into basicsr/models/iet_model.py. This update allows for successful inference on RTX 4090 GPUs without running into memory issues.**
```bash
python basicsr/test.py -opt options/test/001_IET_SRx2_scratch.yml
python basicsr/test.py -opt options/test/002_IET_SRx3_finetune.yml
python basicsr/test.py -opt options/test/003_IET_SRx4_finetune.yml
```

- IET-light (Lightweight Image Super-Resolution)
```bash
python basicsr/test.py -opt options/test/101_IET_light_SRx2_scratch.yml
python basicsr/test.py -opt options/test/102_IET_light_SRx3_finetune.yml
python basicsr/test.py -opt options/test/103_IET_light_SRx4_finetune.yml
```


## <a name="results"></a> 📈 Results
- Classical Image Super-Resolution

<img width="800" src="figures/results_classical.png">

- Lightweight Image Super-Resolution

<img width="800" src="figures/results_light.png">



## <a name="visual-results"></a> 🔎 Visual Results

<img width="800" src="figures/vis_classical.png">

<img width="800" src="figures/vis_light.png">


## <a name="visualazation-of-attention-distributions"></a> 📡 Visualization of Attention Distributions
<img width="800" src="figures/vis_attention_candidates.png">


## <a name="acknowledgements"></a> 💖 Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary.git) and [PFT](https://github.com/LabShuHangGU/PFT-SR.git).

## <a name="citation"></a> 📚 Citation
```bash
@misc{meng2026iet,
      title={From Local Windows to Adaptive Candidates via Individualized Exploratory: Rethinking Attention for Image Super-Resolution}, 
      author={Chunyu Meng and Wei Long and Shuhang Gu},
      year={2026},
      eprint={2601.08341},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.08341}, 
}
```
