def set_segmentation_input_channels(config):
    if "Coco" in config.dataset.name:
        if not config.include_rgb:
            config.in_channels = 2  # just sobel
        else:
            config.in_channels = 3  # rgb
            if config.preprocessor.sobelize:
                config.in_channels += 2  # rgb + sobel
        config.using_IR = False
    elif config.dataset == "Potsdam":
        if not config.include_rgb:
            config.in_channels = 1 + 2  # ir + sobel
        else:
            config.in_channels = 4  # rgbir
            if config.preprocessor.sobelize:
                config.in_channels += 2  # rgbir + sobel
        config.using_IR = True
    else:
        # HACK
        if not config.include_rgb:
            config.in_channels = 1
        else:
            assert False
