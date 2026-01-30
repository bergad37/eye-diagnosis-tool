from timm import create_model

def vit_large_patch16(**kwargs):
    return create_model("vit_large_patch16_224", **kwargs)

def build_retfound_model():
    model = vit_large_patch16(
        num_classes=0,       # remove classification head
        global_pool='avg'    # use valid string, not True
    )
    return model
