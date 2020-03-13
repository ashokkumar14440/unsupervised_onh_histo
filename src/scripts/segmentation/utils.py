import torch

from src.utils.cluster.transforms import sobel_process


IMAGE_1 = 0
IMAGE_2 = 1
AFFINE = 2
MASK = 3


def transfer_images(loader_tuple, config):
    images = create_image_list(config)
    current_batch_size = get_dataloader_batch_size(loader_tuple)
    for index in range(config.num_dataloaders):
        img1, img2, affine2_to_1, mask_img1 = loader_tuple[index]
        assert img1.shape[0] == current_batch_size

        actual_batch_start = index * current_batch_size
        actual_batch_end = actual_batch_start + current_batch_size

        images[IMAGE_1][actual_batch_start:actual_batch_end, :, :, :] = img1
        images[IMAGE_2][actual_batch_start:actual_batch_end, :, :, :] = img2
        images[AFFINE][actual_batch_start:actual_batch_end, :, :] = affine2_to_1
        images[MASK][actual_batch_start:actual_batch_end, :, :] = mask_img1

        # if not (current_batch_size == config.dataloader_batch_sz) and (
        #     e_i == next_epoch
        # ):
        #     print("last batch sz %d" % curr_batch_sz)

        total_size = current_batch_size * config.num_dataloaders  # times 2
        images[IMAGE_1] = images[IMAGE_1][:total_size, :, :, :]
        images[IMAGE_2] = images[IMAGE_2][:total_size, :, :, :]
        images[AFFINE] = images[AFFINE][:total_size, :, :]
        images[MASK] = images[MASK][:total_size, :, :]
    return images


def sobelize(images, config):
    images[IMAGE_1] = sobel_process(
        images[IMAGE_1], config.include_rgb, using_IR=config.using_IR
    )
    images[IMAGE_2] = sobel_process(
        images[IMAGE_2], config.include_rgb, using_IR=config.using_IR
    )
    return images


def process(images, net, head):
    return [net(images[IMAGE_1], head=head), net(images[IMAGE_2], head=head)]


def compute_losses(config, loss_fn, lamb, images, outs):
    # averaging over heads
    avg_loss_batch = None
    avg_loss_no_lamb_batch = None

    for i in range(config.num_sub_heads):
        loss, loss_no_lamb = loss_fn(
            outs[IMAGE_1][i],
            outs[IMAGE_2][i],
            all_affine2_to_1=images[AFFINE],
            all_mask_img1=images[MASK],
            lamb=lamb,
            half_T_side_dense=config.half_T_side_dense,
            half_T_side_sparse_min=config.half_T_side_sparse_min,
            half_T_side_sparse_max=config.half_T_side_sparse_max,
        )

        if avg_loss_batch is None:
            avg_loss_batch = loss
            avg_loss_no_lamb_batch = loss_no_lamb
        else:
            avg_loss_batch += loss
            avg_loss_no_lamb_batch += loss_no_lamb

    avg_loss_batch /= config.num_sub_heads
    avg_loss_no_lamb_batch /= config.num_sub_heads
    return [avg_loss_batch, avg_loss_no_lamb_batch]


def create_image_list(config):
    return [
        create_empty(config),
        create_empty(config),
        create_empty_affine(config),
        create_empty_mask(config),
    ]


def create_empty(config):
    empty = torch.zeros(
        config.batch_sz, get_channel_count(config), config.input_sz, config.input_sz
    )
    return empty.to(torch.float32).cuda()


def create_empty_affine(config):
    empty = torch.zeros(config.batch_sz, 2, 3)
    return empty.to(torch.float32).cuda()


def create_empty_mask(config):
    empty = torch.zeros(config.batch_sz, config.input_sz, config.input_sz)
    return empty.to(torch.float32).cuda()


def get_channel_count(config):
    if not config.no_sobel:
        channel_count = config.in_channels - 1
    else:
        channel_count = config.in_channels
    return channel_count


def get_dataloader_batch_size(loader_tuple):
    return loader_tuple[0][0].shape[0]
