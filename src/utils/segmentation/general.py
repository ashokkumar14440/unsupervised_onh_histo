def set_segmentation_input_channels(config):
    # TODO this entire function is nonsense. The input channels should be
    # identifiable from the data at the point of use.
    ir = False
    rgb = config.dataset.parameters.use_rgb
    sobel = config.preprocessor.sobelize
    if "Coco" in config.dataset.label_filter.name:  # HACK
        assert not ir
        if rgb and sobel:
            config.in_channels = 5  # rgb + sobel
        elif rgb:
            config.in_channels = 3  # rgb
        elif sobel:
            config.in_channels = 3  # gray + sobel
        else:
            config.in_channels = 1  # gray
    else:
        assert not ir
        assert not rgb
        if sobel:
            config.in_channels = 2  # gray + sobel
        else:
            config.in_channels = 1  # gray
