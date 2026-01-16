"""Utility functions."""

import csv
import logging
import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.spatial
import skimage.filters
import skimage.measure
import tifffile
from cellpose.metrics import mask_ious
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border, watershed
from skimage.util import map_array

logger = logging.getLogger(__name__)


def find_images(
    files: list[str],
) -> list[str]:
    """Find the images in the input file or directory paths.

    Adds any file with extension CZI, or any TIFF with 3 dimensions (CYX).

    Args:
        files: Image file or directory paths.

    Returns:
        list of image files

    Raises:
        RuntimeError: if a path is not a file or directory
    """
    images = []
    for fn in files:
        if os.path.isfile(fn):
            if _is_image(fn):
                images.append(fn)
        elif os.path.isdir(fn):
            # List CZI or TIFF
            for file in os.listdir(fn):
                file = os.path.join(fn, file)
                if _is_image(file):
                    images.append(file)
        else:
            raise RuntimeError("Not a file or directory: " + fn)
    return images


def _is_image(fn: str) -> bool:
    """Check if the file is a target image."""
    base, suffix = os.path.splitext(fn)
    suffix = suffix.lower()
    if suffix == ".czi":
        return True
    if suffix == ".tiff":
        # Require CYX. Ignores result image masks (YX) from a directory.
        with tifffile.TiffFile(fn) as tif:
            # image shape
            shape = tif.series[0].shape
            return len(shape) == 3 and np.argmin(shape) == 0
    return False


def find_objects(
    label_image: npt.NDArray[Any],
) -> list[tuple[int, int, tuple[slice, slice]]]:
    """Find the objects in the labeled image.

    Identifies the size and bounding box of objects. The bounding box (bb) is
    a tuple of slices of [min_row, max_row) and [min_col, max_col)
    suitable for extracting the region using im[bb[0], bb[1]].

    This method combines numpy.bincount with scipy.ndimage.find_objects.

    Args:
        label_image: Label image.

    Returns:
        list of (ID, size, (slice(min_row, max_row), slice(min_col, max_col)))
    """
    data = []
    h = np.bincount(label_image.ravel())
    objects = ndi.find_objects(label_image)
    for i, bb in enumerate(objects):
        if bb is None:
            continue
        label = i + 1
        data.append((label, int(h[label]), bb))
    return data


def find_micronuclei(
    label_image: npt.NDArray[Any],
    objects: list[tuple[int, int, tuple[slice, slice]]] | None = None,
    distance: int = 20,
    size: int = 2000,
    min_size: int = 50,
) -> list[tuple[int, int, int, float]]:
    """Find the micro-nuclei objects.

    Identifies all micro-nuclei as objects smaller then the size threshold.
    For each micro-nucleus, searches for an adjacent nucleus within the
    threshold distance to assign as the parent.

    Args:
        label_image: Label image.
        objects: Objects of interest (computed using find_objects).
        distance: Search distance for bleb.
        size: Maximum micro-nucleus size.
        min_size: Minimum micro-nucleus size.

    Returns:
        list of (ID, size, parent ID, distance)
    """
    if objects is None:
        objects = find_objects(label_image)
    sizes = {label: area for (label, area, _) in objects}

    data: list[tuple[int, int, int, float]] = []
    for label, area, bbox in objects:
        if area < min_size:
            continue
        if area > size:
            data.append((label, area, 0, 0.0))
            continue

        # Extract the bounding box plus the search distance
        y, yy, x, xx = (
            max(0, bbox[0].start - distance),
            min(label_image.shape[0], bbox[0].stop + distance),
            max(0, bbox[1].start - distance),
            min(label_image.shape[1], bbox[1].stop + distance),
        )
        crop = label_image[y:yy, x:xx]

        # Check if another object is within the box:
        other = set(np.unique(crop))
        other.remove(label)
        if 0 in other:
            other.remove(0)
        # ignore neighbour micronuclei
        other_list = list(other)
        for parent in other_list:
            if sizes[parent] <= size:
                other.remove(parent)
        if len(other) == 0:
            data.append((label, area, 0, 0.0))
            continue

        # Identify border pixels for each object
        b1 = _find_border(crop, [label])
        b2 = _find_border(crop, list(other))
        # Create KD-tree for the micronuclei and other objects
        c1 = np.argwhere(b1)
        c2 = np.argwhere(b2)
        tree1 = scipy.spatial.KDTree(c1)
        # Find the distance and index of the object in the tree (ignored) for each border pixel
        distances, _index = tree1.query(c2, distance_upper_bound=distance)
        # Get the closest neighbour object (handle ties with a count)
        min_d = distance + 1
        count: dict[int, int] = {}
        for d, c in zip(distances, c2, strict=False):
            if min_d < d:
                continue
            y, x = c
            parent = int(crop[y, x])
            if min_d == d:
                count[parent] = count.get(parent, 0) + 1
            else:
                min_d = d
                count = {parent: 1}

        # Note if multiple parents have the same count of neighbour pixels
        # then choose the largest, otherwise the choice is undefined and
        # defaults to the parent ID
        neighbours = sorted([(v, sizes[k], k) for (k, v) in count.items()])
        if neighbours:
            data.append((label, area, neighbours[0][-1], float(min_d)))
        else:
            data.append((label, area, 0, 0.0))

    return data


def _find_border(
    label_image: npt.NDArray[Any], labels: list[int]
) -> npt.NDArray[Any]:
    """Find border pixels for all labelled objects."""
    mask = np.zeros(label_image.shape, dtype=bool)
    eroded = np.zeros(label_image.shape, dtype=bool)
    strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for label in labels:
        target = label_image == label
        mask = mask | target
        eroded = eroded | ndi.binary_erosion(target, strel)
    border = label_image * mask - label_image * eroded
    return border.astype(label_image.dtype)


def collate_groups(
    data: list[tuple[int, int, int, float]],
) -> list[tuple[int, ...]]:
    """Collate the groups by joining labels with their parent.

    The input data is generated by find_micronuclei.

    Args:
        data: list of (ID, size, parent ID, distance).

    Returns:
        Labels of objects in each group.
    """
    # Initialise so each group contains only itself
    group_d = {x: [x] for (x, *_) in data}
    # Merge labels into their parent
    for label, _, parent, _ in data:
        if parent:
            del group_d[label]
            group_d[parent].append(label)
    return sorted([tuple(x) for x in group_d.values()])


def classify_objects(
    data: list[tuple[int, int, int, float]],
    max_size: int,
    max_distance: float,
) -> dict[int, str]:
    """Classify the objects.

    Classify large objects as nuclei. Small objects are either micro-nuclei,
    or a bleb if closer than the threshold distance to their parent.

    The input data is generated by find_micronuclei.

    Args:
        data: list of (ID, size, parent ID, distance).
        max_size: Maximum micro-nucleus size.
        max_distance: Maximum distance of bleb to the parent nucleus.

    Returns:
        Labels of objects in each group.
    """
    class_names = {}
    for x, size, parent, distance in data:
        if size <= max_size:
            cls = "bleb" if distance < max_distance and parent else "mni"
        else:
            cls = "nucleus"
        class_names[x] = cls
    return class_names


def object_threshold(
    im: npt.NDArray[Any],
    label_image: npt.NDArray[Any],
    fun: Callable[[npt.NDArray[Any]], int],
    objects: list[tuple[int, int, tuple[slice, slice]]] | None = None,
    fill_holes: int = 0,
    min_size: int = 0,
    split_objects: int = 0,
    global_threshold: bool = False,
) -> npt.NDArray[Any]:
    """Threshold the pixels in each masked object.

    The thresholding function accepts a histogram of pixel
    value counts and returns the threshold level above
    which pixels are foreground.

    Args:
        im: Image pixels.
        label_image: Label image.
        objects: Objects of interest (computed using find_objects).
        fun: Thresholding method.
        fill_holes: Remove contiguous holes smaller than the specified size.
        min_size: Minimum size of thresholded regions.
        split_objects: Split objects using a watershed based on: 1=EDT; 2=image.
        global_threshold: Apply thresholding to all object values together; else each object separately.

    Returns:
        mask of thresholded objects
    """
    if objects is None:
        objects = find_objects(label_image)
    final_mask = np.zeros(im.shape, dtype=int)
    total = 0

    if global_threshold:
        data = []
        for label, _area, bbox in objects:
            crop_i = im[bbox]
            crop_m = label_image[bbox]
            target = crop_m == label
            data.append(crop_i[target])
        h = np.bincount(np.concatenate(data))
        t = fun(h)
        logger.info("Global threshold: %d", t)

    for label, _area, bbox in objects:
        # crop for efficiency
        crop_i = im[bbox[0], bbox[1]]
        crop_m = label_image[bbox[0], bbox[1]]
        # threshold the object
        target = crop_m == label
        if not global_threshold:
            values = crop_i[target]
            h = np.bincount(values.ravel())
            t = fun(h)
            logger.debug("Label %d threshold: %d", label, t)
        # create labels
        target = target & (crop_i > t)
        if fill_holes:
            target = skimage.morphology.remove_small_holes(
                target, fill_holes, out=target
            )
        labels, n = skimage.measure.label(target, return_num=True)
        # Watershed to split touching foci
        if split_objects:
            # Watershed based-on distance transform
            distance = (
                ndi.distance_transform_edt(target)
                if split_objects == 1
                else crop_i
            )
            coords = peak_local_max(
                distance, footprint=np.ones((3, 3)), labels=target
            )
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels = watershed(-distance, markers, mask=target)
            n = np.max(labels)
        if min_size > 0:
            labels, n = filter_segmentation(
                labels, border=-1, min_size=min_size
            )
        if total:
            labels[labels != 0] += total
        total += n

        final_mask[bbox[0], bbox[1]] += labels

    return compact_mask(final_mask, m=total)


def compact_mask(mask: npt.NDArray[Any], m: int = 0) -> npt.NDArray[Any]:
    """Compact the int datatype to the smallest required to store all mask IDs.

    Args:
        mask: Segmentation mask.
        m: Maximum value in the mask.

    Returns:
        Compact segmentation mask.
    """
    if m == 0:
        m = mask.max()
    if m < 2**8:
        return mask.astype(np.uint8)
    if m < 2**16:
        return mask.astype(np.uint16)
    return mask


def threshold_method(
    name: str,
    std: float = 4,
    q: float = 0.5,
    threshold: int = 0,
) -> Callable[[npt.NDArray[Any]], int]:
    """Create a threshold function.

    Supported functions:

    mean_plus_std: Threshold using (mean + n * std).
    mean_plus_std_q: Threshold using (mean + n * std) using lowest quantile (q) of values.
    otsu: Otsu thresholding.
    yen: Yen's method.
    minimum: Smoth the histogram until only two maxima and return the mid-point between them.

    Args:
        name: Method name.
        std: Factor n for (mean + n * std) mean_plus_std method.
        q: Quantile for lowest set of values.
        threshold: Manual threshold (overrides named methods).

    Returns:
        Callable threshold method.
    """
    if threshold > 0:

        def manual(h: npt.NDArray[Any]) -> int:
            return threshold

        return manual

    if name == "mean_plus_std":

        def mean_plus_std(h: npt.NDArray[Any]) -> int:
            values = np.arange(len(h))
            probs = h / np.sum(h)
            mean = np.sum(probs * values)
            sd = np.sqrt(np.sum(probs * (values - mean) ** 2))
            t = np.clip(math.floor(mean + std * sd), 0, values[-1])
            return int(t)

        return mean_plus_std

    if name == "mean_plus_std_q":

        def mean_plus_std_q(h: npt.NDArray[Any]) -> int:
            # Find lowest n values to achieve quantile (or next above)
            cumul = np.cumsum(h)
            n = np.searchsorted(cumul / cumul[-1], q) + 1
            values = np.arange(n)
            probs = h[:n] / cumul[n - 1]
            mean = np.sum(probs * values)
            sd = np.sqrt(np.sum(probs * (values - mean) ** 2))
            t = np.clip(math.floor(mean + std * sd), 0, len(h) - 1)
            return int(t)

        return mean_plus_std_q

    if name == "otsu":

        def otsu(h: npt.NDArray[Any]) -> int:
            return int(skimage.filters.threshold_otsu(hist=h))

        return otsu

    if name == "yen":

        def yen(h: npt.NDArray[Any]) -> int:
            return int(skimage.filters.threshold_yen(hist=h))

        return yen

    if name == "minimum":

        def minimum(h: npt.NDArray[Any]) -> int:
            try:
                return int(skimage.filters.threshold_minimum(hist=h))
            except RuntimeError as e:
                print(e)
                return -1

        return minimum

    raise Exception(f"Unknown method: {name}")


def filter_method(
    sigma1: float = 0, sigma2: float = 0
) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]]:
    """Create a filter function.

    The filter will be a Gaussian smoothing filter or a difference of Gaussians
    if the second radius is larger than the first.

    Args:
        sigma1: First Gaussian filter standard deviation.
        sigma2: Second Gaussian filter standard deviation.

    Returns:
        Callable filter method.
    """
    if sigma1 > 0 or sigma2 > sigma1:

        def f(im: npt.NDArray[Any]) -> npt.NDArray[Any]:
            # Foreground smoothing
            im1 = (
                ndi.gaussian_filter(im, sigma1, mode="mirror")
                if sigma1 > 0
                else im
            )

            if sigma2 > sigma1:
                # Background subtraction
                background = ndi.gaussian_filter(im, sigma2, mode="mirror")
                # Do not allow negative values but return as same datatype
                # to support unsigned int images.
                result = im1.astype(np.float64) - background
                im1 = (result - np.min(result)).astype(im.dtype)

            return im1

        return f

    def identity(im: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return im

    return identity


def spot_analysis(
    label_image: npt.NDArray[Any],
    objects: list[tuple[int, int, tuple[slice, slice]]],
    groups: list[tuple[int, ...]],
    im1: npt.NDArray[Any],
    label1: npt.NDArray[Any],
    im2: npt.NDArray[Any],
    label2: npt.NDArray[Any],
    neighbour_distance: float = 20,
) -> list[tuple[int | float, ...]]:
    """Analyse object labels in two image channels within each group of objects.

    The group object is combined using the labels from the provided groups.
    The labels within the two images are compared for overlap of single objects
    and the Intersection-over-Union (IoU) and Mander's coefficient computed.

    Data is returned for each object label within the first and second image:

    group: Group number (from 1).
    channel: Channel of label.
    label: Label.
    parent: Label of parent from the group of objects.
    size: Size of label.
    mean intensity: Mean intensity of label image.
    cx: Centroid x.
    cy: Centroid y.
    overlap: Label from the other channel overlap.
    iou: Intersection-over-Union (IoU) of the overlap with the other channel object.
    m: Mander's coefficient of the overlap.
    neighbour: Label from the other channel neighbour.
    distance: Distance to nearest neighbour.

    Args:
        label_image: Label iamge.
        objects: Objects of interest (computed using find_objects).
        groups: Labels of objects in each group.
        im1: First image.
        label1: First image object labels.
        im2: Second image.
        label2: Second image object labels.
        neighbour_distance: Max distance to nearest neighbour.

    Returns:
        analysis results
    """
    # object look-up by label
    object_d = {o[0]: o for o in objects}

    results: list[tuple[int | float, ...]] = []
    for group_id, group in enumerate(groups):
        group_id += 1
        bbox = object_d[group[0]][2]
        y, yy, x, xx = bbox[0].start, bbox[0].stop, bbox[1].start, bbox[1].stop
        for label in group[1:]:
            bbox = object_d[label][2]
            y, yy, x, xx = (
                min(y, bbox[0].start),
                max(yy, bbox[0].stop),
                min(x, bbox[1].start),
                max(xx, bbox[1].stop),
            )
        # Simplify the group object by cropping and relabel
        mask = _extract_labels(label_image[y:yy, x:xx], group)
        c_label_, id_ = relabel(label_image[y:yy, x:xx] * mask)
        c_label1, id1 = relabel(label1[y:yy, x:xx] * mask)
        c_label2, id2 = relabel(label2[y:yy, x:xx] * mask)
        if len(id1) == 0 and len(id2) == 0:
            # Nothing to analyse
            continue
        c_im1 = im1[y:yy, x:xx]
        c_im2 = im2[y:yy, x:xx]
        # Objects
        objects1 = find_objects(c_label1)
        objects2 = find_objects(c_label2)
        # iou for label1 with label2
        iou1, match1 = mask_ious(c_label1, c_label2)
        # total intensity, cx, cy
        data1 = analyse_objects(c_im1, c_label1, objects1, (y, x))
        data2 = analyse_objects(c_im2, c_label2, objects2, (y, x))
        # Compute Mander's coefficient for the matches
        manders1 = {}
        manders2 = {}
        for i, l2 in enumerate(match1):
            if iou1[i] == 0:
                continue
            l1 = i + 1
            # crop to overlap
            bbox1 = objects1[i][2]
            bbox2 = objects2[l2 - 1][2]
            y, yy, x, xx = (
                min(bbox1[0].start, bbox2[0].start),
                max(bbox1[0].stop, bbox2[0].stop),
                min(bbox1[1].start, bbox2[1].start),
                max(bbox1[1].stop, bbox2[1].stop),
            )
            mask = (c_label1[y:yy, x:xx] == l1) & (c_label2[y:yy, x:xx] == l2)
            manders1[l1] = float(
                (c_im1[y:yy, x:xx] * mask).sum() / data1[i][0]
            )
            manders2[l2] = float(
                (c_im2[y:yy, x:xx] * mask).sum() / data2[l2 - 1][0]
            )
        # Create reverse IoU lookup
        iou2 = np.zeros(len(data2))
        match2 = np.zeros(len(data2), dtype=np.int_)
        for i, (iou, m) in enumerate(zip(iou1, match1, strict=True)):
            if iou:
                j = m - 1
                match2[j] = i + 1
                iou2[j] = iou
        # closest neighbour distance matching
        d1, d2, neighbour1, neighbour2 = (
            np.full(len(data1), -1.0),
            np.full(len(data2), -1.0),
            np.zeros(len(data1), dtype=np.int_),
            np.zeros(len(data2), dtype=np.int_),
        )
        if len(data1) and len(data2):
            c1 = np.array(data1)[:, -2:]
            c2 = np.array(data2)[:, -2:]
            row_ind, col_ind, cost = _map_partial_linear_sum(
                c1, c2, neighbour_distance
            )
            for r, c in zip(row_ind, col_ind, strict=True):
                d1[r] = d2[c] = cost[r, c]
                neighbour1[r] = c
                neighbour2[c] = r
        # Report
        for (
            ch_,
            data_,
            id_,
            other_id_,
            objects_,
            iou_,
            match_,
            manders_,
            d_,
            neighbour_,
        ) in zip(
            [1, 2],
            [data1, data2],
            [id1, id2],
            [id2, id1],
            [objects1, objects2],
            [iou1, iou2],
            [match1, match2],
            [manders1, manders2],
            [d1, d2],
            [neighbour1, neighbour2],
            strict=True,
        ):
            for i, d in enumerate(data_):
                cx, cy = d[1], d[2]
                parent = int(label_image[math.floor(cy), math.floor(cx)])
                results.append(
                    (
                        group_id,
                        ch_,
                        int(id_[i]),
                        parent,
                        objects_[i][1],
                        d[0] / objects_[i][1],
                        cx,
                        cy,
                        int(other_id_[match_[i] - 1]) if iou_[i] else 0,
                        float(iou_[i]),
                        manders_.get(i + 1, 0),
                        int(other_id_[neighbour_[i]]) if d_[i] >= 0 else 0,
                        float(d_[i]),
                    )
                )

    return results


def _extract_labels(
    label_image: npt.NDArray[Any], labels: tuple[int, ...]
) -> npt.NDArray[Any]:
    """Extract a mask for all labelled objects."""
    mask = np.zeros(label_image.shape, dtype=bool)
    for label in labels:
        target = label_image == label
        mask = mask | target
    return mask


def filter_segmentation(
    mask: npt.NDArray[Any],
    border: int = 5,
    relabel: bool = True,
    min_size: int = 10,
) -> tuple[npt.NDArray[Any], int]:
    """Removes border objects and filters small objects from segmentation mask.

    The size of the border can be specified. Use a negative size to skip removal of
    border objects.

    Objects are optionally relabeled to be continuous from 1.
    Note: Removal of border objects can result in missing labels from the
    original [0, N] label IDs.

    Args:
        mask: unfiltered segmentation mask
        border: width of the border examined (negative to disable)
        relabel: Set to True to relabel objects in [0, N]
        min_size: Minimum size of objects.

    Returns:
        filtered segmentation mask, number of objects (N)
    """
    cleared: npt.NDArray[Any] = (
        mask if border < 0 else clear_border(mask, buffer_size=border)  # type: ignore[no-untyped-call]
    )
    sizes = np.bincount(cleared.ravel())
    mask_sizes = sizes > min_size
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    new_mask = cells_cleaned * mask
    n = np.sum(mask_sizes)
    if relabel:
        old_id = np.arange(len(sizes))[mask_sizes]
        new_mask = map_array(
            new_mask, old_id, np.arange(1, 1 + n), out=np.zeros_like(mask)
        )
    return new_mask, n  # type: ignore[no-any-return]


def relabel(
    mask: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], list[int]]:
    """Relabels the mask to be continuous from 1.

    The old id array contains the original ID for each new label:

    original id = old_id[label - 1]

    Args:
        mask: unfiltered segmentation mask

    Returns:
        relabeled mask, old_id
    """
    sizes = np.bincount(mask.ravel())
    sizes[0] = 0
    mask_sizes = sizes != 0
    old_id = np.arange(len(sizes))[mask_sizes]
    n = len(old_id)
    new_mask = map_array(
        mask, old_id, np.arange(1, 1 + n), out=np.zeros_like(mask)
    )
    return new_mask, old_id  # type: ignore[no-any-return]


def analyse_objects(
    im: npt.NDArray[Any],
    label_image: npt.NDArray[Any],
    objects: list[tuple[int, int, tuple[slice, slice]]],
    offset: tuple[int, int] = (0, 0),
) -> list[tuple[float, float, float]]:
    """Extract the intensity and centroids for all labelled objects.

    Centroids use (0.5, 0.5) as the centre of pixels.

    Args:
        im: Image.
        label_image: Label image.
        objects: List of (ID, size, (slice(min_row, max_row), slice(min_col, max_col))).
        offset: Offset to add to the centre (Y,X).

    Returns:
        list of (intensity, cx, cy)
    """
    data = []
    for label, _, bbox in objects:
        crop_image = im[bbox[0], bbox[1]]
        crop_label = label_image[bbox[0], bbox[1]]
        mask = crop_label == label
        intensity = float((crop_image * mask).sum())
        y, x = np.nonzero(mask)
        weights = crop_image[mask].ravel()
        weights = weights / weights.sum()
        cy, cx = (
            float(np.sum(y * weights) + bbox[0].start + offset[0] + 0.5),
            float(np.sum(x * weights) + bbox[1].start + offset[1] + 0.5),
        )
        data.append((intensity, cx, cy))

    return data


def _map_partial_linear_sum(
    c1: npt.NDArray[Any], c2: npt.NDArray[Any], threshold: float
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[Any]]:
    """Minimum weight bipartite graph matching using partial cost matrix of Euclidean distance.

    Args:
        c1: Coordinates 1
        c2: Coordinates 1
        threshold: Distance threshold for alignment mappings

    Returns:
        Mapping from row -> column, partial cost matrix
    """
    # Dense matrix built using KD-Trees with some false edges.
    tree1 = scipy.spatial.KDTree(c1)
    tree2 = scipy.spatial.KDTree(c2)
    # Ensure a full matching exists by setting false edges between all vertices.
    # Use a distance that cannot be chosen over an actual edge.
    cost = np.full((len(c1), len(c2)), len(c1) * threshold * 1.5)
    indexes = tree1.query_ball_tree(tree2, r=threshold)
    count = 0
    for i, v1 in enumerate(c1):
        cm = cost[i]
        count += len(indexes[i])
        # Note: If there are no indexes then the threshold is too low.
        # We could: (a) Increase the threshold until there some edges for
        # all vertices; (b) choose n random points, find their closest
        # neighbours and use to estimate the threshold. Currently the
        # vertex should join a false edge and be removed later.
        for j in indexes[i]:
            v2 = c2[j]
            d = v1 - v2
            cm[j] = np.sqrt((d * d).sum())
    row_ind, col_ind = linear_sum_assignment(cost)

    # Ignore large distance mappings. Can occur due to false edges.
    for i, (r, c) in enumerate(zip(row_ind, col_ind, strict=True)):
        if cost[r][c] > threshold:
            row_ind[i], col_ind[i] = -1, -1
    selected = row_ind >= 0
    row_ind = row_ind[selected]
    col_ind = col_ind[selected]
    return row_ind, col_ind, cost


def spot_summary(
    results: list[tuple[int | float, ...]],
    groups: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Summarise the spots results by group.

    group: Group number.
    label: Parent label.
    count1: Number of spots in channel 1.
    count2: Number of spots in channel 2.

    Args:
        results: Spot analysis results.
        groups: Labels of objects in each group.

    Returns:
        summary
    """
    # Count of spots in each parent label
    count1: dict[int, int] = {}
    count2: dict[int, int] = {}
    for _group, ch, _label, parent, *_ in results:
        parent = int(parent)
        d = count1 if ch == 1 else count2
        d[parent] = d.get(parent, 0) + 1

    summary: list[tuple[int, ...]] = []
    for group_id, group in enumerate(groups):
        group_id += 1
        for label in group:
            summary.append(
                (group_id, label, count1.get(label, 0), count2.get(label, 0))
            )
    return summary


def format_spot_results(
    results: list[tuple[int | float, ...]],
    class_names: dict[int, str] | None = None,
    scale: float = 0,
) -> list[tuple[int | float | str, ...]]:
    """Format the spot results.

    If a scale is provided an additional scaled distance column is returned.
    Units are assumed to be micrometers (μm).

    Args:
        results: Spot analysis results.
        class_names: Optional class name of each object.
        scale: Optional distance scale (micrometers/pixel).

    Returns:
        Formatted results
    """
    out: list[tuple[int | float | str, ...]] = []
    out.append(
        (
            "group",
            "channel",
            "label",
            "parent",
            "class",
            "size",
            "mean",
            "cx",
            "cy",
            "overlap",
            "iou",
            "manders",
            "neighbour",
            "distance",
        )
    )
    if scale:
        out[0] = out[0] + ("distance (μm)",)
    for data in results:
        parent = int(data[3])
        cls = class_names.get(parent, "") if class_names else ""
        formatted = data[:4] + (cls,) + data[4:]
        if scale:
            formatted = formatted + (data[-1] * scale,)
        out.append(formatted)

    return out


def format_summary_results(
    summary: list[tuple[int, ...]],
    class_names: dict[int, str] | None = None,
    object_data: dict[int, tuple[int, float, float, float]] | None = None,
) -> list[tuple[int | float | str, ...]]:
    """Format the spot summary results.

    Args:
        summary: Spot analysis summary results.
        class_names: Optional class name of each object.
        object_data: Optional data for each object (size, intensity, cx, cy)

    Returns:
        Formatted results
    """
    out: list[tuple[int | float | str, ...]] = []
    out.append(
        (
            "group",
            "label",
            "size",
            "intensity",
            "cx",
            "cy",
            "class",
            "count1",
            "count2",
            "total",
        )
    )
    for data in summary:
        parent = int(data[1])
        parent_data = (
            object_data.get(parent, (0, 0, 0, 0))
            if object_data
            else (0, 0, 0, 0)
        )
        cls = class_names.get(parent, "") if class_names else ""
        formatted = (
            data[:2] + parent_data + (cls,) + data[2:] + (data[-2] + data[-1],)
        )
        out.append(formatted)

    return out


def save_csv(
    fn: str,
    data: list[Any],
) -> None:
    """Save the data to a CSV file.

    Args:
        fn: File name.
        data: Data.
    """
    with open(fn, "w", newline="") as f:
        csv.writer(f).writerows(data)
