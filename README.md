[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10838384.svg)](https://doi.org/10.5281/zenodo.10838384)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10838432.svg)](https://doi.org/10.5281/zenodo.10838432)
[![DOI](http://img.shields.io/badge/DOI-TO.BE.UPDATED.SOON-B31B1B)](https://TO.BE.UPDATED.SOON)

# AI-driven segmentation of microvascular features  of tissue-engineered vascular grafts

<a name="contents"></a>
## 📖 Contents
- [Introduction](#introduction)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Access](#data-access)
- [How to Cite](#how-to-cite)


<a name="introduction"></a>
## 🎯 Introduction
This repository presents an artificial intelligence (AI)-driven approach for the precise segmentation and quantification of histological features observed during the microscopic examination of tissue-engineered vascular grafts (TEVGs). The development of next-generation TEVGs is a leading trend in translational medicine, offering minimally invasive surgical interventions and reducing the long-term risk of device failure. However, the analysis of regenerated tissue architecture poses challenges, necessitating AI-assisted tools for accurate histological evaluation.

<a name="data"></a>
## 📁 Data
The study utilized a dataset comprising 104 Whole Slide Images (WSIs) obtained from biodegradable TEVGs implanted into the carotid arteries of 20 sheep. After six months, the sheep were euthanized to assess vascular tissue regeneration patterns. The WSIs were automatically sliced into 99,831 patches, which underwent filtering and manual annotation by pathologists. A total of 1,401 patches were annotated, identifying nine histological features: _arteriole lumen_, _arteriole media_, _arteriole adventitia_, _venule lumen_, _venule wall_, _capillary lumen_, _capillary wall_, _immune cells_, and _nerve trunks_ (<a href="#figure-1">Figure 1</a>). These annotations were meticulously verified by a senior pathologist, ensuring accuracy and consistency.

<p align="center">
  <img id="figure-1" width="80%" height="80%" src=".assets/annotation_methodology.png" alt="Annotation methodology">
</p>

<p align="left">
    <em><strong>Figure 1.</strong> Annotation methodology for histology patches (top row) depicting features associated with a blood vessel regeneration (replacement of a biodegradable polymer by de novo formed vascular tissue). Histological annotations delineated with segmentation masks (bottom row) include arteriole lumen (red), arteriole media (pink), arteriole adventitia (light pink), venule lumen (blue), venule wall (light blue), capillary lumen (brown), capillary wall (tan), immune cells (lime), and nerve trunks (yellow).</em>
</p>


<a name="methods"></a>
## 🔬 Methods
The methodology involved two main stages: hyperparameter tuning and model training. Six deep learning models ([U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28), [LinkNet](https://ieeexplore.ieee.org/document/8305148), [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf), [PSPNet](https://arxiv.org/abs/1612.01105), [DeepLabV3](https://arxiv.org/abs/1706.05587), and [MA-Net](https://ieeexplore.ieee.org/document/9201310)) were rigorously tuned across 200 configurations to achieve optimal performance. Hyperparameters such as encoder architecture, input image size, optimizer, and learning rate were extensively explored using Bayesian optimization and [HyperBand](https://arxiv.org/abs/1603.06560) early termination strategies.

Following the tuning stage, the models were trained and evaluated on the entire dataset using a 5-fold cross-validation approach (<a href="#figure-2">Figure 2</a>). This ensured the integrity of subject groups within each subset, preventing data leakage. During training, various augmentation techniques were applied to expand the dataset and mitigate overfitting.

<p align="center">
  <img id="figure-2" width="80%" height="80%" src=".assets/loss_evolution.png" alt="Loss and DSC evolution">
</p>

<p align="left">
    <em><strong>Figure 2.</strong> Comparative analysis of loss and DSC evolution during training and testing phases over 5-fold cross-validation with 95% confidence interval.</em>
</p>

<a name="results"></a>
## 📈 Results


<a name="conclusion"></a>
## 🏁 Conclusion


<a name="requirements"></a>
## 💻 Requirements
- Operating System
  - [x] macOS
  - [x] Linux
  - [x] Windows (limited testing carried out)
- Python 3.11.x
- Required core libraries: [environment.yaml](https://github.com/ViacheslavDanilov/histology_segmentation/blob/main/environment.yaml)

<a name="installation"></a>
## ⚙ Installation
**Step 1:** Install Miniconda

Installation guide: https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install

**Step 2:** Clone the repository and change the current working directory
``` bash
git clone https://github.com/ViacheslavDanilov/histology_segmentation.git
cd histology_segmentation
```

**Step 3:** Set up an environment and install the necessary packages
``` bash
chmod +x make_env.sh
./make_env.sh
```

<a name="data-access"></a>
## 🔐 Data Access
All essential components of the study, including the curated dataset and trained models, have been made publicly available:
- **Dataset:** [https://doi.org/10.5281/zenodo.10838384](https://doi.org/10.5281/zenodo.10838384)
- **Models:** [https://doi.org/10.5281/zenodo.10838432](https://doi.org/10.5281/zenodo.10838432)

<a name="how-to-cite"></a>
## 🖊️ How to Cite
Please cite [our paper](https://TO.BE.UPDATED.SOON) if you found our data, methods, or results helpful for your research:

> Danilov V.V., et al. (**2024**). _AI-driven segmentation of microvascular features during histological examination of tissue-engineered vascular grafts_. **Frontiers in Cell and Developmental Biology**. DOI: [TO.BE.UPDATED.SOON](TO.BE.UPDATED.SOON)
