from pathlib import Path, PurePath

import torch

import architecture as arch
import utils


def setup(config):
    # INPUT IMAGE INFORMATION
    image_info = utils.ImageInfo(**config.dataset.parameters)

    # ARCH HEAD INFORMATION
    heads_info = arch.HeadsInfo(
        heads_info=config.architecture.heads.info,
        input_size=config.dataset.parameters.input_size,
        subhead_count=config.architecture.heads.subhead_count,
    )

    # OUTPUT_FILES
    output_root = PurePath(config.output.root) / str(config.dataset.id)
    output_root = PurePath(Path(output_root).resolve())
    if not (Path(output_root).is_dir() and Path(output_root).exists()):
        Path(output_root).mkdir(parents=True, exist_ok=True)
    output_files = utils.OutputFiles(root_path=output_root, image_info=image_info)

    # STATE_FOLDER
    state_folder = output_files.get_sub_root(output_files.STATE)

    # RENDERING PATHS
    # TODO into output_files
    dataset = PurePath(config.dataset.root)
    if "partitions" in config.dataset:
        partitions = config.dataset.partitions
        image_folder = dataset / partitions.image
        label_folder = dataset / partitions.label
    else:
        image_folder = dataset
        label_folder = None

    # NETWORK ARCHITECTURE
    structure = arch.Structure(
        input_channels=image_info.channel_count,
        structure=config.architecture.trunk.structure,
    )
    trunk = arch.VGGTrunk(structure=structure, **config.architecture.trunk.parameters)
    net = arch.SegmentationNet10aTwoHead(
        trunk=trunk, heads=heads_info.build_heads(trunk.feature_count)
    )
    net.to(torch.device("cuda:0"))

    return {
        "image_info": image_info,
        "heads_info": heads_info,
        "output_files": output_files,
        "state_folder": state_folder,
        "image_folder": image_folder,
        "label_folder": label_folder,
        "net": net,
    }
