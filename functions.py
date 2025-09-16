import pyclesperanto_prototype as cle
import numpy as np
from scipy import ndimage
from skimage import exposure, img_as_float32

def resample_isotropic(volume, voxel_size=(2.0, 0.5, 0.5), target_iso=None, order=1):
    """
    Resample a 3D volume to isotropic voxel spacing.
    - volume: ndarray (Z, Y, X)
    - voxel_size: tuple (z, y, x) in microns (or whatever units)
    - target_iso: desired isotropic voxel size; if None uses min(voxel_size)
    - order: interpolation order (0..5)
    Returns resampled volume (float32).
    """
    if target_iso is None:
        target_iso = min(voxel_size)
    z_scale = voxel_size[0] / target_iso
    y_scale = voxel_size[1] / target_iso
    x_scale = voxel_size[2] / target_iso
    zoom = (z_scale, y_scale, x_scale)
    out = ndimage.zoom(volume.astype(np.float32), zoom=zoom, order=order)
    return out

def resample_isotropic_gpu(volume, voxel_size=(2.0, 0.5, 0.5), target_iso=None):
    """
    Resample a 3D volume to isotropic voxel spacing using pyclesperanto (GPU).
    - volume: ndarray (Z, Y, X)
    - voxel_size: tuple (z, y, x) in microns (or other units)
    - target_iso: desired isotropic voxel size; if None, uses min(voxel_size)
    Returns resampled volume as numpy.ndarray (float32).
    """
    if target_iso is None:
        target_iso = min(voxel_size)

    # Compute scaling factors (old_spacing / new_spacing)
    z_scale = voxel_size[0] / target_iso
    y_scale = voxel_size[1] / target_iso
    x_scale = voxel_size[2] / target_iso

    # Push to GPU
    gpu_vol = cle.push(volume.astype(np.float32))

    # Scale isotropically
    gpu_iso = cle.scale(gpu_vol, factor_x=x_scale, factor_y=y_scale, factor_z=z_scale, linear_interpolation=True)

    # Pull back to CPU
    return cle.pull(gpu_iso)



def normalize_clahe(vol, kernel_size, clip_limit=0.1, slicewise=False):
    """
    Apply CLAHE (skimage.exposure.equalize_adapthist) to a 3D input image.

    Parameters
    ----------
    vol : ndarray
        3D image array (Z, Y, X)
    kernel_size: tuple
    clip_limit : float
        CLAHE clip limit
    slicewise : bool
        If True, apply CLAHE slice by slice (2D); otherwise full 3D

    Returns
    -------
    ndarray
        CLAHE-processed image in float32, range [0,1]

    Reference
    ---------
    https://scikit-image.org/docs/0.24.x/auto_examples/color_exposure/plot_adapt_hist_eq_3d.html
    """
    
    # Rescale image data to range [0, 1]
    # vol = np.clip(vol, np.percentile(vol, 5), np.percentile(vol, 95))
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    
    if slicewise:
        # Apply CLAHE on each Z slice separately (2D)
        out = np.zeros_like(vol, dtype=np.float32)
        for z in range(vol.shape[0]):
            out[z] = exposure.equalize_adapthist(vol[z],kernel_size=kernel_size, clip_limit=clip_limit)
    else:
        # Apply full 3D CLAHE
        out = exposure.equalize_adapthist(vol, kernel_size=kernel_size, clip_limit=clip_limit)

    return out.astype(np.float32)


def normalize_global_zscore(vol, percentile=(1, 99), eps=1e-8, method="meanstd", rescale=True, clip_range=(-3, 3)):
    """
    Normalize a volume using z-score normalization.

    Steps:
    1) Clip intensities to given percentiles (robust to outliers).
    2) Apply chosen z-score method:
       - "meanstd": (I - mean) / std
       - "medianmad": (I - median) / MAD (robust to outliers)
    3) Optionally rescale to [0,1] by mapping clip_range to 0..1.

    Parameters
    ----------
    vol : ndarray
        Input volume (2D or 3D).
    percentile : tuple of (low, high)
        Percentile range for clipping (default: (1,99)).
    eps : float
        Small value to prevent division by zero.
    method : str
        "meanstd" (classic) or "medianmad" (robust).
    rescale : bool
        If True, map values from clip_range to [0,1].
    clip_range : tuple (low, high)
        Range of z-scores to map to [0,1] (default: -3..+3).

    Returns
    -------
    z : ndarray (float32)
        Z-score normalized (and optionally rescaled) volume.
    """

    # p_low, p_high = np.percentile(vol, percentile)
    # v = np.clip(vol, p_low, p_high).astype(np.float32)
    v = vol
    if method == "meanstd":
        mu = v.mean()
        sigma = v.std() + eps
        z = (v - mu) / sigma
    
    elif method == "medianmad":
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + eps
        z = (v - med) / mad
    
    else:
        raise ValueError(f"Unknown method '{method}', use 'meanstd' or 'medianmad'.")
    
    if rescale:
        low, high = clip_range
        z = (z - low) / (high - low)
        z = np.clip(z, 0.0, 1.0)
    
    return z.astype(np.float32)




def normalize_global_zscore_gpu(vol, percentile=(1, 99), eps=1e-8):
    """
    Global z-score normalization with percentile clipping on GPU.
    Percentiles are computed on CPU.
    """
    # compute percentiles on CPU
    p_low, p_high = np.percentile(vol, percentile)
    
    # push to GPU
    gpu_img = cle.push(vol.astype(np.float32))
    
    # clip values on GPU
    gpu_img = cle.clip(gpu_img, p_low, p_high)
    
    # mean and std on GPU
    mu = cle.mean_of_all_pixels(gpu_img)
    sigma = cle.standard_deviation_of_all_pixels(gpu_img) + eps
    
    # global z-score
    gpu_img = cle.subtract_image_and_scalar(gpu_img, mu)
    gpu_img = cle.divide_image_and_scalar(gpu_img, sigma)
    
    # map -3..3 to 0..1
    gpu_img = cle.add_image_and_scalar(gpu_img, 3.0)
    gpu_img = cle.multiply_image_and_scalar(gpu_img, 1/6.0)
    gpu_img = cle.clip(gpu_img, 0.0, 1.0)
    
    return cle.pull(gpu_img)

