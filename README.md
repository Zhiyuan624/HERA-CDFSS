# [CVPR 2026] Selective, Regularized, and Calibrated: Harnessing Vision Foundation Models for Cross-Domain Few-Shot Semantic Segmentation

Official implementation of our CVPR 2026 paper [Selective, Regularized, and Calibrated: Harnessing Vision Foundation Models for Cross-Domain Few-Shot Semantic Segmentation](https://arxiv.org/abs/2605.19340)

<p align="center">
  <strong>
    <a href="https://arxiv.org/abs/2605.19340">Paper</a> |
    <a href="https://zhiyuan624.github.io/HERA-CDFSS/">Project Page</a> |
    <a href="https://github.com/Zhiyuan624/HERA-CDFSS">Code</a>
  </strong>
</p>

<p align="center">
  <img src="assets/method.jpg" width="100%">
</p>

The proposed **HERA** harnesses vision foundation models for cross-domain few-shot semantic segmentation through selective feature extraction, regularized adaptation, and calibrated attention refinement. It dynamically identifies task-relevant representations from intermediate layers, integrates complementary multi-level features, and improves prediction reliability under severe domain shifts while requiring only a few annotated support examples.

## Data Preparation
We evaluate HERA on the standard cross-domain few-shot semantic segmentation benchmarks. 

The source-domain dataset follows the conventional CD-FSS setting, while HERA is evaluated directly on the target domains without source-data retraining.

### Source Domain

#### PASCAL VOC 2012
PASCAL VOC 2012 is commonly used as the source-domain dataset in CD-FSS.

- **Official Website:** [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- **Train/Val Archive:**

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

- **SDS Extended Mask Annotations:** [Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)

---

### Target Domains

#### DeepGlobe
DeepGlobe is a satellite-image segmentation dataset with substantial variations in texture, scale, and spatial layout.

- **Official Website:** [DeepGlobe](http://deepglobe.org/)
- **Download:** [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)

#### ISIC 2018
ISIC 2018 contains dermoscopic skin-lesion images with irregular boundaries and low-contrast foreground regions.

- **Official Website:** [ISIC 2018 Challenge](http://challenge2018.isic-archive.com/)
- **Download:** [ISIC Archive](https://challenge.isic-archive.com/data#2018)  
  Registration and login may be required.
- **Class Information:** `data/isic/class_id.csv`
- **Preprocessing References:** [PATNet repository](https://github.com/slei109/PATNet)

#### Chest X-ray
The Chest X-ray dataset contains radiographic images and lung masks, introducing substantial grayscale and structural domain shifts.

- **Dataset Description:** [NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)
- **Download:** [Kaggle](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)

#### FSS-1000
FSS-1000 is a large-scale few-shot segmentation dataset containing 1,000 object categories from natural images.

- **Official Repository:** [FSS-1000](https://github.com/HKUSTCV/FSS-1000)
- **Download:** [Google Drive](https://drive.google.com/file/d/16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI/view)
