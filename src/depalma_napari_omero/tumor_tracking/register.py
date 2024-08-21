# from depalma_napari_omero.omero_widget import combine_images

import napari
import numpy as np
import skimage.io
from skimage.measure import marching_cubes
from skimage.filters import gaussian
import scipy.ndimage as ndi
from vedo import Points

# from mouselungseg import LungsPredictor, extract_3d_roi


def register_timeseries(timeseries, lungs_timeseries, order=3):
    """Registers a timeseries dataset based on the lungs data."""
    image0 = timeseries[0]
    registered_timeseries = np.empty_like(timeseries)
    registered_timeseries[0] = image0

    lung0 = lungs_timeseries[0]
    registered_lungs_timeseries = np.empty_like(lungs_timeseries)
    registered_lungs_timeseries[0] = lung0

    for k, (image1, lung1) in enumerate(zip(timeseries[1:], lungs_timeseries[1:])):
        Phi = fit_affine_from_lungs_masks(lung0, lung1)

        warped_image1 = apply_transform(image1, Phi, order=order)
        registered_timeseries[k + 1] = warped_image1

        warped_lung1 = apply_transform(lung1, Phi, order=0)
        registered_lungs_timeseries[k + 1] = warped_lung1

    return registered_timeseries, registered_lungs_timeseries


def fit_affine_from_lungs_masks(lungs0, lungs1):
    """Estimates an affine transformation matrix that brings lung1 onto lung0 from two sets of corresponding masks."""
    verts0, *_ = marching_cubes(gaussian(lungs0.astype(float), sigma=1), level=0.5)

    verts1, *_ = marching_cubes(gaussian(lungs1.astype(float), sigma=1), level=0.5)

    aligned_pts1 = (
        Points(verts1).clone().align_to(Points(verts0), invert=True, use_centroids=True)
    )

    Phi = aligned_pts1.transform.matrix

    return Phi


def apply_transform(image, Phi, order: int = 3):
    """Applies an affine transform to warp a 3D image."""
    warped = ndi.affine_transform(
        image, matrix=Phi[:3, :3], offset=Phi[:3, 3], order=order
    )

    return warped


def combine_images(*images):
    """Inserts images at different times, of different shapes, into a single (TZYX) array."""
    n_images = len(images)
    image_shapes = np.stack([np.array(img.shape) for img in images])
    output_shape = [n_images]
    output_shape.extend(list(np.max(image_shapes, axis=0)))

    timeseries = np.empty(output_shape, dtype=images[0].dtype)
    for k, (image, image_shape) in enumerate(zip(images, image_shapes)):
        delta = (output_shape[1:] - image_shape) // 2
        timeseries[k][
            delta[0] : delta[0] + image_shape[0],
            delta[1] : delta[1] + image.shape[1],
            delta[2] : delta[2] + image.shape[2],
        ] = image

    return timeseries


if __name__ == "__main__":

    # predictor = LungsPredictor()

    # im0_original = skimage.io.imread('/home/wittwer/code/amaia/depalma/depalma-napari-omero/notebooks/test_data/C30827/T02.tif')
    # im0_lungs_original = predictor.predict(im0_original)
    # im0, im0_lungs = extract_3d_roi(im0_original, im0_lungs_original)

    # im1_original = skimage.io.imread('/home/wittwer/code/amaia/depalma/depalma-napari-omero/notebooks/test_data/C30827/T03.tif')
    # im1_lungs_original = predictor.predict(im1_original)
    # im1, im1_lungs = extract_3d_roi(im1_original, im1_lungs_original)

    im0 = skimage.io.imread(
        "/home/wittwer/code/amaia/depalma/depalma-napari-omero/notebooks/test_data/C30827/T02-roi.tif"
    )
    im0_lungs = skimage.io.imread(
        "/home/wittwer/code/amaia/depalma/depalma-napari-omero/notebooks/test_data/C30827/T02-roi-lungs.tif"
    )

    im1 = skimage.io.imread(
        "/home/wittwer/code/amaia/depalma/depalma-napari-omero/notebooks/test_data/C30827/T03-roi.tif"
    )
    im1_lungs = skimage.io.imread(
        "/home/wittwer/code/amaia/depalma/depalma-napari-omero/notebooks/test_data/C30827/T03-roi-lungs.tif"
    )

    Phi = fit_affine_from_lungs_masks(im0_lungs, im1_lungs)
    warped_im1 = apply_transform(im1, Phi)
    warped_im1_lungs = apply_transform(im1_lungs, Phi, order=0)

    viewer = napari.Viewer()

    # viewer.add_image(im0)
    # viewer.add_labels(im0_lungs)

    # viewer.add_image(im1)
    # viewer.add_labels(im1_lungs)

    # viewer.add_image(warped_im1)
    # viewer.add_labels(warped_im1_lungs)

    ### Time series
    # timeseries = combine_images(im0, im1)
    # timeseries_lungs = combine_images(im0_lungs, im1_lungs)
    # timeseries_lungs_warped = combine_images(im0_lungs, warped_im1_lungs)

    im0_tumors = skimage.io.imread(
        "/home/wittwer/code/amaia/depalma/depalma-napari-omero/notebooks/test_data/C30827/T02-roi_mask.tif"
    )
    im1_tumors = skimage.io.imread(
        "/home/wittwer/code/amaia/depalma/depalma-napari-omero/notebooks/test_data/C30827/T03-roi_mask.tif"
    )
    im1_tumors_warped = apply_transform(im1_tumors, Phi, order=0)

    from depalma_napari_omero.tumor_tracking import (
        run_tracking,
        remap_timeseries_labels,
    )

    timeseries = combine_images(im0_tumors, im1_tumors)
    linkage_df, grouped_df = run_tracking(timeseries)
    timeseries = remap_timeseries_labels(timeseries, linkage_df)
    viewer.add_labels(timeseries)

    timeseries_warped = combine_images(im0_tumors, im1_tumors_warped)
    linkage_df, grouped_df = run_tracking(timeseries_warped)
    timeseries_warped = remap_timeseries_labels(timeseries_warped, linkage_df)
    viewer.add_labels(timeseries_warped)

    napari.run()
