"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

try:
    from itertools import izip_longest as zip_longest  # Python 2
except ImportError:
    from itertools import zip_longest

import os
import sys
import tensorboardX
import shutil
import time

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "10.8.2.250"  # configure this to your master node's IP
    os.environ["MASTER_PORT"] = "29500"  # configure this to your master node's port
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size: int, total_epochs: int, save_every: int, opts: argparse.Namespace) -> None:
    ddp_setup(rank, world_size)

    torch.cuda.empty_cache()
    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path

    # Setup model and data loader
    if opts.trainer == 'MUNIT':
        trainer = MUNIT_Trainer(config, segmentation=opts.segmentation)
    elif opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")

    gpu_id = rank
    trainer.cuda(gpu_id)
    trainer = DDP(trainer, device_ids=[gpu_id])

    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).to(gpu_id)
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).to(gpu_id)
    test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).to(gpu_id)
    test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).to(gpu_id)

    print("Training images A: %d, B: %d. Testing images A: %d, B: %d" % (
        len(train_loader_a.dataset), len(train_loader_b.dataset), len(test_loader_a.dataset), len(test_loader_b.dataset)))

    data_len = len(train_loader_a)

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path, model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # Start training
    iterations = trainer.module.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0

    total_time = 0
    torch.cuda.synchronize()
    start_time = time.time()
    while True:
        # assert len(train_loader_a) == len(train_loader_b), "Data loaders must have the same number of batches"
        for it, (images_a, images_b) in enumerate(zip_longest(train_loader_a, train_loader_b, fillvalue=None)):
            train_loader_a.sampler.set_epoch(it % data_len)
            train_loader_b.sampler.set_epoch(it % data_len)

            if images_a is None or images_b is None:
                print("Skiped iteration")
                continue

            try:
                images_a, images_b = images_a.to(gpu_id).detach(), images_b.to(gpu_id).detach()

                with Timer("Elapsed time in update: %f"):
                    # Main training code
                    trainer.module.dis_update(images_a, images_b, config)
                    trainer.module.gen_update(images_a, images_b, config)
                    torch.cuda.synchronize()

            except Exception as e:
                print(e)
                continue

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                torch.cuda.synchronize()
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer.module, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.module.sample(test_display_images_a.to(gpu_id), test_display_images_b.to(gpu_id))
                    train_image_outputs = trainer.module.sample(train_display_images_a.to(gpu_id), train_display_images_b.to(gpu_id))
                torch.cuda.synchronize()
                print("Saving images at iteration %d" % (iterations + 1))
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.module.sample(train_display_images_a.to(gpu_id), train_display_images_b.to(gpu_id))
                torch.cuda.synchronize()
                print("Saving display images at iteration %d" % (iterations + 1))
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if ((iterations + 1) % config['snapshot_save_iter'] == 0 and gpu_id == 0):
                print("Saving models at iteration %d" % (iterations + 1))
                trainer.module.save(checkpoint_directory, iterations)
            trainer.module.update_learning_rate()
            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            print("Total time: %02d:%02d:%02d" % (total_time // 3600, total_time % 3600 // 60, total_time % 60))

        destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
    parser.add_argument('--config', type=str, default='configs/tir2rgb_folder.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    parser.add_argument('--segmentation', action="store_true")
    opts = parser.parse_args()

    config = get_config(opts.config)

    total_epochs = 3
    save_every = config['snapshot_save_iter']
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, save_every, opts), nprocs=world_size)
