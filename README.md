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
within SSL data is beneficial for model performance.

## Models
The models used in this study are based on the [MetaFormer](https://arxiv.org/abs/2210.13452) architecture. The models are trained using a self-supervised learning approach on the SurgeNet
dataset, introduced this [paper](https://). All model weights can be downloaded from the table below.

| Model           | Epochs | Teacher Backbone                                                                                                                          | Full DINO checkpoint                                                                                                                 |
|-----------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| SurgeNet        | 25     | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0025_teacher.pth?download=true)      | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint0025.pth?download=true) |
| SurgeNet-Small  | 50     | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint_epoch0050_teacher.pth?download=true) | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint0050.pth?download=true) |
| SurgeNet-CHOLEC | 50     | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/CHOLEC_checkpoint_epoch0050_teacher.pth?download=true)        | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/CHOLEC_checkpoint0050.pth?download=true) | 
| SurgeNet-RAMIE  | 50     | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RAMIE_checkpoint_epoch0050_teacher.pth?download=true)         | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RAMIE_checkpoint0050.pth?download=true) | 
| SurgeNet-RARP   | 50     | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RARP_checkpoint_epoch0050_teacher.pth?download=true)          | [Download](https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RARP_checkpoint0050.pth?download=true) |


## Loading Model
The weights from the teacher network can be used to initialize either your classification or segmentation model using the following code snippet:

```python
import torch
from MetaFormer import caformer_s18, MetaFormerFPN

# load the weights
weights = torch.load('path/to/weights.pth')

# Just the backbone
classification_model = caformer_s18()
classification_model.load_state_dict(weights)

# Full segmentation model
segmentation_model = MetaFormerFPN(weights=weights)

```

## Acknowledgements


## Citation
If you find our work useful in your research please consider citing our paper: