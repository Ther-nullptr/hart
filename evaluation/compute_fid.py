"""
Improved FID computation for HART model evaluation
Refactored with better structure, modularity and MJHQ-30K integration
"""
import json
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from utils import tracker

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path)
            assert img.mode == "RGB"
            if self.transforms is not None:
                img = self.transforms(img)
        except Exception as e:
            raise FileNotFoundError(path, "\n", e)

        return img


def get_activations(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, img_size=512):
    model.eval()

    if batch_size > len(files):
        print("Warning: batch size is bigger than the data size. Setting batch size to data size")
        batch_size = len(files)
    
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])
    
    dataset = ImagePathDataset(files, transforms=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )

    pred_arr = np.empty((len(files), dims))
    start_idx = 0

    for batch in tqdm(dataloader, desc="Computing FID activations"):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx : start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, img_size=512):
    act = get_activations(files, model, batch_size, dims, device, num_workers, img_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1, img_size=512):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size, dims, device, num_workers, img_size)

    return m, s


def calculate_fid_given_paths(paths, batch_size=50, device='cpu', dims=2048, num_workers=1, img_size=512):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size, dims, device, num_workers, img_size)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size, dims, device, num_workers, img_size)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--num-workers', type=int, default=1,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--img-size', type=int, default=1024,
                        help='Image size for evaluation')
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths to the generated images or '
                              'to .npz statistic files'))
    parser.add_argument('--exp_name', type=str, default="experiment",
                        help='Experiment name for logging')
    parser.add_argument('--report_to', type=str, default="wandb",
                        help='Where to report metrics')
    parser.add_argument('--tracker_project_name', type=str, default="hart-evaluation",
                        help='Project name for tracking')

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, see
            # https://bugs.python.org/issue33581
            num_cpus = os.cpu_count()

        args.num_workers = min(8, num_cpus)

    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          args.num_workers,
                                          args.img_size)
    print(f'FID: {fid_value}')

    # Log to tracker if specified
    if args.report_to == "wandb":
        result_dict = {args.exp_name: fid_value}
        tracker(args, result_dict, label="", pattern="epoch_step", metric="FID")


if __name__ == '__main__':
    main()