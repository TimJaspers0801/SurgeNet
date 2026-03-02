import torch
import timm

# This code loads all DINO model weights (v1, v2, v3) using timm library. Some unexpected keys are ignored during loading, this is intended behaviour.

# Optionally specify classes if needed for classifcation tasks
n_classes = 10 

# URLs for pre-trained DINO weights
urls = {
    'path_dinov1_vits': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv1_ViTs16_size224_SurgeNetXL.pth?download=true',
    'path_dinov1_vitb': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv1_ViTb16_size224_SurgeNetXL.pth?download=true',

    'path_dinov2_vits': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTs14_size336_SurgeNetXL.pth?download=true',
    'path_dinov2_vitb': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTb14_size336_SurgeNetXL.pth?download=true',
    'path_dinov2_vitl': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTl14_size336_SurgeNetXL.pth?download=true',

    'path_dinov3_vits': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTs16_size336_SurgeNetXL.pth?download=true',
    'path_dinov3_vitb': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTb16_size336_SurgeNetXL.pth?download=true',
    'path_dinov3_vitl': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTl16_size336_SurgeNetXL.pth?download=true',
}


###################################
### Loading dinov1(using timm) ###
###################################
 
# ViT-s
model = timm.create_model(
    'vit_small_patch16_224.dino',
    img_size=(224, 224),
    patch_size=16,
    num_classes=n_classes,
)
state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov1_vits'])
msg = model.load_state_dict(state_dict, strict=False)
print("\nLoaded DINOv1 ViT-s weights with msg:\n", msg)

# ViT-b
model = timm.create_model(
    'vit_base_patch16_224.dino',
    img_size=(224, 224),
    patch_size=16,
    num_classes=n_classes,
)
state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov1_vitb'])
msg = model.load_state_dict(state_dict, strict=False)
print("\nLoaded DINOv1 ViT-b weights with msg:\n", msg)


###################################
### Loading dinov2 (using timm) ###
###################################

# ViT-s
model = timm.create_model(
    'vit_small_patch14_dinov2',
    img_size=(336, 336),
    patch_size=14,
    num_classes=n_classes,
)
state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov2_vits'])
msg = model.load_state_dict(state_dict, strict=False)
print("\nLoaded DINOv2 ViT-s weights with msg:\n", msg)

# ViT-b
model = timm.create_model(
    'vit_base_patch14_dinov2',
    img_size=(336, 336),
    patch_size=14,
    num_classes=n_classes,
)
state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov2_vitb'])
msg = model.load_state_dict(state_dict, strict=False)
print("\nLoaded DINOv2 ViT-b weights with msg:\n", msg)

# ViT-l
model = timm.create_model(
    'vit_large_patch14_dinov2',
    img_size=(336, 336),
    patch_size=14,
    num_classes=n_classes,
)
state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov2_vitl'])
msg = model.load_state_dict(state_dict, strict=False)
print("\nLoaded DINOv2 ViT-l weights with msg:\n", msg)


###########################################
### Loading dinov3 (using transformers) ###
###########################################

# ViTs
model = timm.create_model(
    'vit_small_patch16_dinov3.lvd1689m',
    img_size=(336, 336),
    patch_size=16,
    num_classes=n_classes,
)
state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov3_vits'])
msg = model.load_state_dict(state_dict, strict=False)
print("\nLoaded DINOv3 ViT-s weights with msg:\n", msg)

# ViTb
model = timm.create_model(
    'vit_base_patch16_dinov3.lvd1689m',
    img_size=(336, 336),
    patch_size=16,
    num_classes=n_classes,
)
state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov3_vitb'])
msg = model.load_state_dict(state_dict, strict=False)
print("\nLoaded DINOv3 ViT-b weights with msg:\n", msg)
 
# ViTl
model = timm.create_model(
    'vit_large_patch16_dinov3.lvd1689m',
    img_size=(336, 336),
    patch_size=16,
    num_classes=n_classes,
)
state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov3_vitl'])
msg = model.load_state_dict(state_dict, strict=False)
print("\nLoaded DINOv3 ViT-l weights with msg:\n", msg)
