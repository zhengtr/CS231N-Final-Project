import segmentation_models_pytorch as smp


def get_models(model_name='unet'):
    
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            decoder_use_batchnorm=True
    )
    elif model_name == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            decoder_use_batchnorm=True
    )
    elif model_name.lower() == 'fpn':
        model = smp.FPN(
            encoder_name="resnext101_32x8d",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            decoder_dropout=0.2
    )
    return model