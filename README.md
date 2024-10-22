# SurgeNet
![FIG 1.](figures/SurgeNet.png)

## Publications
This repository contains code for the models used in the following publications:

- [Tim J.M. Jaspers *et al.*](https://arxiv.org/abs/2407.17904) - Exploring the Effect of Dataset Diversity in
Self-Supervised Learning for Surgical Computer
Vision (*accepted - Data Engineering in Medical Imaging (DEMI) - Satellite Event MICCAI 2024*)

  
## Abstract
Over the past decade, computer vision applications in minimally invasive surgery have rapidly increased. Despite this growth, the
impact of surgical computer vision remains limited compared to other medical fields like pathology and radiology, primarily due to the scarcity
of representative annotated data. Whereas transfer learning from large
annotated datasets such as ImageNet has been conventionally the norm
to achieve high-performing models, recent advancements in self-supervised
learning (SSL) have demonstrated superior performance. In medical image analysis, in-domain SSL pretraining has already been shown to outperform ImageNet-based initialization. Although unlabeled data in the
field of surgical computer vision is abundant, the diversity within this data is limited. This study investigates the role of dataset diversity in
SSL for surgical computer vision, comparing procedure-specific datasets against a more heterogeneous general surgical dataset across three different downstream surgical applications. 
The obtained results show that using solely procedure-specific data can lead to substantial improvements of 13.8%, 9.5%, and 36.8% compared to ImageNet pretraining. However, extending this data with more heterogeneous surgical data fur-
ther increases performance by 5.0%, 5.2%, and 2.5%, suggesting that increasing diversity within SSL data is beneficial for model performance.

## Results
Figure below shows the performance of the models trained on different datasets for the three downstream dataset each from a unique procedure. 
![FIG 2.](figures/Results_SurgeNet.svg)

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
from metaformer import caformer_s18, MetaFormerFPN

urls = {
    "ImageNet1k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth",
    "SurgeNet": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0025_teacher.pth?download=true",
    "SurgeNet-Small": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-CHOLEC": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/CHOLEC_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-RAMIE": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RAMIE_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-RARP": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RARP_checkpoint_epoch0050_teacher.pth?download=true"
}

# classification model
classification_model = caformer_s18(num_classes=12, pretrained='SurgeNet', pretrained_weights=urls['SurgeNet'])

# Full segmentation model
segmentation_model = MetaFormerFPN(num_classes=4, pretrained='SurgeNet', pretrained_weights=urls['SurgeNet'])

```
Note: If your want a different version of SurgeNet weights (e.g. SurgeNet-Small), you can replace the `pretrained_weights` argument with the desired url (leave the `pretrained` argument as it is).
## Acknowledgements
Our implementation of the feature pyramid network is based on the [pytorch segmentation models](https://segmentation-modelspytorch.readthedocs.io/en/latest/) library.
Pretraining on SurgeNet was performed using the code provided with the [DINO](https://github.com/facebookresearch/dino) publication.

## Citation
If you find our work useful in your research please consider citing our paper:
```bibtex
@misc{jaspers2024datasetdiversity,
      title={Exploring the Effect of Dataset Diversity in Self-Supervised Learning for Surgical Computer Vision}, 
      author={Tim J. M. Jaspers and Ronald L. P. D. de Jong and Yasmina Al Khalil and Tijn Zeelenberg and Carolus H. J. Kusters and Yiping Li and Romy C. van Jaarsveld and Franciscus H. A. Bakker and Jelle P. Ruurda and Willem M. Brinkman and Peter H. N. De With and Fons van der Sommen},
      year={2024},
      eprint={2407.17904},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17904}, 
}
```