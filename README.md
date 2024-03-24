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
The study utilized a dataset comprising 104 Whole Slide Images (WSIs) obtained from biodegradable TEVGs implanted into the carotid arteries of 20 sheep. After six months, the sheep were euthanized to assess vascular tissue regeneration patterns. The WSIs were automatically sliced into 99,831 patches, which underwent filtering and manual annotation by pathologists. A total of 1,401 patches were annotated, identifying nine histological features: _arteriole lumen_, _arteriole media_, _arteriole adventitia_, _venule lumen_, _venule wall_, _capillary lumen_, _capillary wall_, _immune cells_, and _nerve trunks_. These annotations were meticulously verified by a senior pathologist, ensuring accuracy and consistency.

<a name="methods"></a>
## 🔬 Methods


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
