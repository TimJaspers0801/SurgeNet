import torch
from metaformer import caformer_s18, MetaFormerFPN
from convnextv2 import convnextv2_tiny, ConvNextFPN
from pvtv2 import pvt_v2_b2, PVTV2FPN

urls = {
    "ImageNet1k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth",
    "SurgeNetXL": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetXL_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-Small": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-CHOLEC": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/CHOLEC_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-RAMIE": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RAMIE_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-RARP": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RARP_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-Public": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/Public_checkpoint0050.pth?download=true",
    "SurgeNet-ConvNextv2": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_ConvNextv2_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-PVTv2": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_PVTv2_checkpoint_epoch0050_teacher.pth?download=true",
}

# CAFormer model
classification_model = caformer_s18(num_classes=12, pretrained='SurgeNet', pretrained_weights=urls['SurgeNetXL'])
segmentation_model = MetaFormerFPN(num_classes=12, pretrained='SurgeNet', pretrained_weights=urls['SurgeNetXL'])

# ConvNextv2 model
classification_model = convnextv2_tiny(num_classes=12, pretrained_weights=urls['SurgeNet-ConvNextv2'])
segmentation_model = ConvNextFPN(num_classes=12, pretrained_weights=urls['SurgeNet-ConvNextv2'])

# PVTv2 model
classification_model = pvt_v2_b2(num_classes=12, pretrained_weights=urls['SurgeNet-PVTv2'])
segmentation_model = PVTV2FPN(num_classes=12, pretrained_weights=urls['SurgeNet-PVTv2'])
