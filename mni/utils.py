"""Utility functions."""

import csv
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.spatial
import skimage.filters
import skimage.measure
from cellpose.metrics import mask_ious
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.util import map_array


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

    Returns:
        mask of thresholded objects
    """
    if objects is None:
        objects = find_objects(label_image)
    final_mask = np.zeros(im.shape, dtype=int)
    total = 0
    for label, _area, bbox in objects:
        # crop for efficiency
        crop_i = im[bbox[0], bbox[1]]
        crop_m = label_image[bbox[0], bbox[1]]
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

        final_mask[bbox[0], bbox[1]] += labels

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
        c1 = np.array(data1)[:, -2:]
        c2 = np.array(data2)[:, -2:]
        row_ind, col_ind, cost = _map_partial_linear_sum(
            c1, c2, neighbour_distance
        )
        d1, d2, neighbour1, neighbour2 = (
            np.full(len(c1), -1.0),
            np.full(len(c2), -1.0),
            np.zeros(len(c1), dtype=np.int_),
            np.zeros(len(c2), dtype=np.int_),
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
                        int(id_[match_[i] - 1]) if iou_[i] else 0,
                        float(iou_[i]),
                        manders_.get(i + 1, 0),
                        int(id_[neighbour_[i]]) if d_[i] >= 0 else 0,
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
    cost = np.full((len(c1), len(c2)), len(c1) * threshold * 1.0)
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
) -> list[tuple[int | float | str, ...]]:
    """Format the spot results.

    Args:
        results: Spot analysis results.
        class_names: Optional class name of each object.

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
    for data in results:
        parent = int(data[3])
        cls = class_names.get(parent, "") if class_names else ""
        formatted = data[:4] + (cls,) + data[4:]
        out.append(formatted)

    return out


def format_summary_results(
    summary: list[tuple[int, ...]],
    class_names: dict[int, str] | None = None,
) -> list[tuple[int | str, ...]]:
    """Format the spot summary results.

    Args:
        summary: Spot analysis summary results.
        class_names: Optional class name of each object.

    Returns:
        Formatted results
    """
    out: list[tuple[int | str, ...]] = []
    out.append(
        (
            "group",
            "label",
            "class",
            "count1",
            "count2",
            "total",
        )
    )
    for data in summary:
        parent = int(data[1])
        cls = class_names.get(parent, "") if class_names else ""
        formatted = data[:2] + (cls,) + data[2:] + (data[-2] + data[-1],)
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
