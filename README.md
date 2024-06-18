# SurgeNet
![FIG 5.](figures/SurgeNet.png)

## Publications
This repository contains code for the models used in the following publications:

- [Tim J.M. Jaspers *et al.*](https://) - Exploring the Effect of Dataset Diversity in
Self-Supervised Learning for Surgical Computer
Vision (*Data Engineering in Medical Imaging (DEMI) - Satellite Event MICCAI 2024*)

  
## Abstract
Over the past decade, computer vision applications in minimally invasive surgery have rapidly increased. Despite this growth, the
impact of surgical computer vision remains limited compared to other
medical fields like pathology and radiology, primarily due to the scarcity
of representative annotated data. While transfer learning from large annotated datasets such as ImageNet has traditionally been the norm to
achieve high-performing models, recent advancements in self-supervised
learning (SSL) have demonstrated superior performance. In medical image analysis, in-domain SSL pretraining has already been shown to out-
perform ImageNet-based initialization. Although unlabeled data in the
field of surgical computer vision is abundant, the diversity within this
data is limited. This study investigates the role of dataset diversity in
SSL for surgical computer vision, comparing procedure-specific datasets
against a more heterogeneous surgical dataset across three different downstream applications.
Our results show that using solely procedure-specific
data can lead to improvements of 13.8%, 9.5%, and 36.8% compared to ImageNet
pretraining. Extending this data with more heterogeneous surgical data further
increased performance by 5.0%, 5.2%, and 2.5%, suggesting that increasing diversity
within SSL data is beneficial for modelperformance.

## Models
The models used in this study are based on the [MetaFormer](https://) architecture. The models are trained using a self-supervised learning approach on the [SurgeNet](https://)
dataset, introduced in [paper](https://).

## Citation
If you find our work useful in your research please consider citing our paper: