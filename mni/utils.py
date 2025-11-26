"""Utility functions."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.spatial
import skimage.filters
import skimage.measure
from scipy import ndimage as ndi


def find_micronuclei(
    im: npt.NDArray[Any],
    distance: int = 20,
    size: int = 2000,
    min_size: int = 50,
) -> list[tuple[int, int, int, float]]:
    """Find the micro-nuclei objects.

    Identifies all micro-nuclei as objecvts smaller then the size threshold.
    For each micro-nucleus, searches for an adjacent nucleus within the
    threshold distance to assign as the parent.

    Args:
        im: Image mask.
        distance: Search distance for bleb.
        size: Maximum micro-nucleus size.
        min_size: Minimum micro-nucleus size.

    Returns:
        list of (ID, size, parent ID, distance)
    """
    # Size of each object
    props = skimage.measure.regionprops(im)
    sizes = {p.label: p.area for p in props}

    # For each micro-nucleus
    data: list[tuple[int, int, int, float]] = []
    for p in props:
        label, area = p.label, int(p.area)
        if p.area < min_size:
            continue
        if p.area > size:
            data.append((label, area, 0, 0.0))
            continue

        # Extract the bounding box plus the search distance
        y, yy, x, xx = (
            max(0, p.bbox[0] - distance),
            min(im.shape[0], p.bbox[2] + distance),
            max(0, p.bbox[1] - distance),
            min(im.shape[1], p.bbox[3] + distance),
        )
        crop = im[y:yy, x:xx]

        # Check if another object is within the box:
        other = set(np.unique(crop))
        other.remove(label)
        if 0 in other:
            other.remove(0)
        # ignore neighbour micronuclei
        for parent in other:
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

        data.append((label, area, neighbours[0][-1], float(min_d)))

    return data


def _find_border(im: npt.NDArray[Any], labels: list[int]) -> npt.NDArray[Any]:
    """Find border pixels for all labelled objects."""
    mask = np.zeros(im.shape, dtype=bool)
    eroded = np.zeros(im.shape, dtype=bool)
    strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for label in labels:
        target = im == label
        mask = mask | target
        eroded = eroded | ndi.binary_erosion(target, strel)
    border = im * mask - im * eroded
    return border.astype(im.dtype)


def object_threshold(
    im: npt.NDArray[Any],
    mask: npt.NDArray[Any],
    fun: Callable[[npt.NDArray[Any]], int],
) -> npt.NDArray[Any]:
    """Threshold the pixels in each masked object.

    The thresholding function accepts a histogram of pixel
    value counts and returns the threshold level above
    which pixels are foreground.

    Args:
        im: Image pixels.
        mask: Image mask.
        fun: Thresholding method.

    Returns:
        mask of thresholded objects
    """
    final_mask = np.zeros(im.shape, dtype=int)
    total = 0
    for i, sl in enumerate(ndi.find_objects(mask)):
        if sl is None:
            continue
        label = i + 1
        # crop for efficiency
        crop_i = im[sl[0], sl[1]]
        crop_m = mask[sl[0], sl[1]]
        # threshold the object
        target = crop_m == label
        values = crop_i[target]
        h = np.bincount(values.ravel())
        t = fun(h)
        # create labels
        target = target & (crop_i > t)
        labels, n = skimage.measure.label(target, return_num=True)
        if total:
            labels[labels != 0] += total
        total += n

        final_mask[sl[0], sl[1]] += labels

    return _compact_mask(final_mask)


def _compact_mask(mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Compact the int datatype to the smallest required to store all mask IDs.

    Args:
        mask (npt.NDArray[Any]): Segmentation mask.

    Returns:
        npt.NDArray[Any]: Compact segmentation mask.
    """
    m = mask.max()
    if m < 2**8:
        return mask.astype(np.uint8)
    if m < 2**16:
        return mask.astype(np.uint16)
    return mask


def threshold_method(
    name: str, std: float = 4
) -> Callable[[npt.NDArray[Any]], int]:
    """Create a threshold function.

    Supported functions:

    mean_plus_std: Threshold using (mean + n * std)
    otsu: Otsu thresholding.
    yen: Yen's method.
    minimum: Smoth the histogram until only two maxima and return the mid-point between them.

    Args:
        name: Method name.
        std: Factor n for (mean + n * std) mean_plus_std method.

    Returns:
        Callable threshold method.
    """
    if name == "mean_plus_std":

        def mean_plus_std(h: npt.NDArray[Any]) -> int:
            values = np.arange(len(h))
            probs = h / np.sum(h)
            mean = np.sum(probs * values)
            sd = np.sqrt(np.sum(probs * (values - mean) ** 2))
            t = np.clip(math.floor(mean + std * sd), 0, values[-1])
            return int(t)

        return mean_plus_std

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
