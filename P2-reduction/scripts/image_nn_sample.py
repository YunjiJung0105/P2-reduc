"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import save_image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

@th.no_grad()
def main():
    args = create_argparser().parse_args()

    args.prefix = os.path.basename(args.model_path).replace('.', '_')

    dist_util.setup_dist()
    logger.configure(dir=args.sample_dir, prefix=args.prefix)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    #if args.use_fp16:
    #    model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    count = 0

    # Added for loading real images!
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=256,
        class_cond=args.class_cond,
        deterministic=True,
        random_flip=False
    )

    real_images = None

    i = 0
    while i<10:
        batch = next(data)
        real_images = batch[0].clone() if real_images is None else th.cat((real_images, batch[0]), 0)
        i+=1

    save_image(real_images, os.path.join(logger.get_dir(), "real.jpg".format(count)), normalize=True, nrow=10)

    real_images = real_images.to(dist_util.dev())
    real_images_org = real_images.clone()

    # ImageNet normalize
    mean, std = th.Tensor([0.485, 0.456, 0.406]).to(dist_util.dev()), th.Tensor([0.229, 0.224, 0.225]).to(dist_util.dev())
    mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
    vgg16_model = th.hub.load("pytorch/vision:v0.6.0", "vgg16", pretrained=True).to(dist_util.dev())
    vgg16_model.classifier = vgg16_model.classifier[:-1]
    vgg16_model.eval()

    real_images = (real_images - real_images.min()) / (real_images.max() - real_images.min())
    real_images = (real_images - mean) / std
    real_images_resize = F.interpolate(real_images, (224, 224), mode='bilinear', align_corners=False)

    real_features = vgg16_model(real_images_resize).squeeze()

    padding = th.zeros(3, 256, 256//10).to(dist_util.dev())

    while count * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 200),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        # get Nearest Neighbors
        sample_for_nn = (sample - sample.min()) / (sample.max() - sample.min())
        sample_for_nn = (sample_for_nn - mean) / std
        sample_for_nn = F.interpolate(sample_for_nn, (224, 224), mode='bilinear', align_corners=False)
        sample_features = vgg16_model(sample_for_nn).squeeze()
        # sample_features = vgg16_model(real_images_resize[:10]).squeeze()

        sample_with_nn_batch = []
        for bidx in range(len(sample_features)):
            distance = th.square(sample_features[bidx] - real_features).mean(dim=1)
            top_k_idx = th.argsort(distance)[:args.n_topk]
            top_k_imgs = real_images_org[top_k_idx]
            flatten = th.cat([sample[bidx]]+[padding]+list(top_k_imgs), 2)
            sample_with_nn_batch.append(flatten)
        
        sample_with_nn_batch = th.stack(sample_with_nn_batch, 0)

        out_path = os.path.join(logger.get_dir(), "{:05d}.jpg".format(count))
        save_image(sample_with_nn_batch, out_path, nrow=1, normalize=True)

        # saving npz
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
        count += 1

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        data_dir="../data/obama/100/",
        num_samples=10000,
        batch_size=10,
        use_ddim=False,
        model_path="",
        sample_dir="",
        prefix="",
        n_topk=7,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
