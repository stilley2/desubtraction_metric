import pandas as pd
import pydicom
import pydicom.errors
import pydicom.uid
import numpy as np
import skimage
import skimage.feature
import skimage.transform

try:
    from . import _version
    __version__ = _version.version
except ImportError:
    __version__ = "0.0.0"


ROI = 250  # mm
RADIUS = 12.5  # mm
HOUGH_MIN_DISTANCE = 5  # mm
CANNY_SIGMA = 1  # mm
NUM_FEATURES = 10  # mm
MASK_RADIUS_FRACTION_FOR_ORDERING = 0.5
MASK_INNER_RADIUS_FRACTION = 0.8
MASK_OUTER_RADIUS_FRACTION = 1.2
PMMA_MID_INDEX = 7
AL_MID_INDEX = 2


def read_phantom():
    phantom = np.array([
                        [-62.5466040683311, -0.28671780863608404, 0.5, 0],
                        [-36.55623920764044, 32.711745690953585, 1.0, 0],
                        [-12.549675498793937, -0.29496311114721413, 1.5, 0],
                        [11.459630483012036, 32.71930991609118, 2.0, 0],
                        [37.44565274645893, -0.29597198220987325, 2.5, 0],
                        [37.45365892552708, -42.308708996638636, 10.0, 1],
                        [11.458209375406152, -75.30952128448895, 8.0, 1],
                        [-12.549642184022852, -42.315128638316224, 6.0, 1],
                        [-36.53880813528695, -75.30898119650237, 4.0, 1],
                        [-62.55791278497006, -42.308066812820734, 2.0, 1]
                        ])
    return phantom


def hough_wrapper(img, pixel_spacing):
    edge = skimage.feature.canny(img, sigma=CANNY_SIGMA / pixel_spacing)
    radius_mid = RADIUS / pixel_spacing
    radii = np.linspace(radius_mid * 0.9, radius_mid * 1.1, 11)
    hough_res = skimage.transform.hough_circle(edge, radii)
    hough_peaks = skimage.transform.hough_circle_peaks(hough_res,
                                                       radii,
                                                       total_num_peaks=NUM_FEATURES,
                                                       min_xdistance=int(HOUGH_MIN_DISTANCE / pixel_spacing),
                                                       min_ydistance=int(HOUGH_MIN_DISTANCE / pixel_spacing))
    return np.stack(hough_peaks[1:3], axis=1)


def get_means(img, pixel_spacing, centers):
    ygrid, xgrid = np.mgrid[:img.shape[0], :img.shape[1]]
    rad = RADIUS / pixel_spacing * MASK_RADIUS_FRACTION_FOR_ORDERING
    means = [np.mean(img[(ygrid - c[1]) ** 2 + (xgrid - c[0])**2 < rad ** 2])
             for c in centers]
    return np.array(means)


def quad_detrend(img, pixel_spacing, centers):
    ygrid, xgrid = np.mgrid[:img.shape[0], :img.shape[1]]
    rad = RADIUS / pixel_spacing * MASK_OUTER_RADIUS_FRACTION
    mask = np.ones(img.shape, dtype=bool)
    for cx, cy in centers:
        mask[(ygrid - cy) ** 2 + (xgrid - cx) ** 2 < rad ** 2] = 0
    X = xgrid[mask]
    Y = ygrid[mask]
    Z = img[mask].astype(np.float64)
    A = np.stack((X ** 2, Y ** 2, X * Y, X, Y, np.ones_like(X)), axis=1).astype(np.float64)
    p = np.linalg.solve(A.transpose() @ A, A.transpose() @ Z)
    X = xgrid.ravel()
    Y = ygrid.ravel()
    A = np.stack((X ** 2, Y ** 2, X * Y, X, Y), axis=1).astype(np.float64)
    return img - (A @ p[:-1]).reshape(img.shape)


def sort_circles(img, pixel_spacing, centers):
    means = get_means(img, pixel_spacing, centers)
    theta = np.arctan2(centers[:, 1] - np.mean(centers[:, 1]), centers[:, 0] - np.mean(centers[:, 0]))
    min_from_center = np.min(np.abs(centers - np.mean(centers, axis=0)), axis=0)
    if min_from_center[0] > min_from_center[1]:
        # split vertically
        group1 = centers[:, 0] < np.mean(centers[:, 0])
        theta = theta - np.pi / 2
        theta[theta < -np.pi] += np.pi * 2
    else:
        # split horizontally
        group1 = centers[:, 1] < np.mean(centers[:, 1])
    group2 = np.logical_not(group1)
    means1_sorted = means[group1][np.argsort(theta[group1])]
    means2_sorted = means[group2][np.argsort(theta[group2])]
    if all(np.diff(means1_sorted) < 0):
        assert(all(np.diff(means2_sorted) > 0))
        aluminum = group1
        pmma = group2
    elif all(np.diff(means1_sorted) > 0):
        assert(all(np.diff(means2_sorted) < 0))
        aluminum = group2
        pmma = group1
    else:
        raise RuntimeError("Unable to determine material")
    return np.concatenate((centers[aluminum][np.argsort(theta[aluminum])], centers[pmma][np.argsort(theta[pmma])]))


def transform_phantom(centers, phantom, pixel_spacing):
    phantom = phantom.copy()
    tradj = skimage.transform.AffineTransform(scale=[pixel_spacing, -pixel_spacing],
                                              translation=np.mean(centers, axis=0) * np.array([-1, 1]) * pixel_spacing)
    centers_adj = tradj(centers)
    tr = skimage.transform.SimilarityTransform()
    if not tr.estimate(phantom[:, :2], centers_adj[:, :2]):
        raise RuntimeError("Unable to estimate transform")
    phantom[:, :2] = tradj.inverse(tr(phantom[:, :2]))
    radius = RADIUS / pixel_spacing * tr.scale
    return phantom, radius


def get_labels(imgshape, circle_data, r1, r2, r3):
    inner = np.zeros(imgshape, dtype='u8')
    outer = np.zeros(imgshape, dtype='u8')
    ygrid, xgrid = np.mgrid[:imgshape[0], :imgshape[1]]
    for i in range(circle_data.shape[0]):
        m1 = (xgrid - circle_data[i, 0]) ** 2.0 + (ygrid - circle_data[i, 1]) ** 2 < r1 ** 2
        if np.any(inner[m1] > 0):
            raise RuntimeError("Masks overlap")
        inner[m1] = i + 1
        m2 = np.logical_xor((xgrid - circle_data[i, 0]) ** 2.0 + (ygrid - circle_data[i, 1]) ** 2 < r2 ** 2,
                            (xgrid - circle_data[i, 0]) ** 2.0 + (ygrid - circle_data[i, 1]) ** 2 < r3 ** 2)
        if np.any(outer[m2] > 0):
            raise RuntimeError("Masks overlap")
        outer[m2] = i + 1
    return inner, outer


def get_w(hi, li, minner, mouter):
    # return np.log(np.mean(hi[mouter]) / np.mean(hi[minner])) / np.log(np.mean(li[mouter]) / np.mean(li[minner]))
    return np.log(np.mean(hi[minner]) / np.mean(hi[mouter])) / np.log(np.mean(li[minner]) / np.mean(li[mouter]))


def cnr(img, inner, outer, nlabels, feature_inds_, feature_mats_, feature_heights, mat, kerma):
    out = {"Feature Index": [],
           "CNR / √X": [],
           "Feature Material": [],
           "Feature Height (mm)": [],
           "DE Image": [],
           "mean inner": [],
           "mean outer": [],
           "var inner": [],
           "var outer": []}
    for i in range(nlabels):
        im = inner == i + 1
        om = outer == i + 1
        out["mean inner"].append(np.mean(img[im]))
        out["mean outer"].append(np.mean(img[om]))
        out["var inner"].append(np.var(img[im]))
        out["var outer"].append(np.var(img[om]))
        out["CNR / √X"].append(np.abs(out["mean inner"][-1] - out["mean outer"][-1]) /
                               np.sqrt(out["var inner"][-1] + out["var outer"][-1]) /
                               np.sqrt(kerma))
        out["Feature Index"].append(feature_inds_[i])
        out["Feature Material"].append(feature_mats_[i])
        out["Feature Height (mm)"].append(feature_heights[i])
        out["DE Image"].append(mat)
    return pd.DataFrame(out)


def load_dcm(f):
    try:
        dataset = pydicom.dcmread(f)
    except (AttributeError, pydicom.errors.InvalidDicomError):
        f.seek(0)
        pydicom.filereader.read_preamble(f, False)
        h = pydicom.filereader.read_dataset(f, is_implicit_VR=False, is_little_endian=True,
                                            stop_when=lambda tag, vr, val: tag.group > 2)
        ts = h["TransferSyntaxUID"]
        if ts.value == pydicom.uid.ExplicitVRBigEndian:
            ivr = False
            le = False
        elif ts.value == pydicom.uid.ExplicitVRLittleEndian:
            ivr = False
            le = True
        elif ts.value == pydicom.uid.ImplicitVRLittleEndian:
            ivr = True
            le = True
        else:
            raise RuntimeError("unsupported transfer sytanx")
        dataset = pydicom.filereader.read_dataset(f, is_implicit_VR=ivr, is_little_endian=le)
        dataset.file_meta = h
    return dataset


def _proc(high_data, low_data, air_kerma, quad_detrend_all):
    high = load_dcm(high_data)
    high_img = high.pixel_array.astype(np.float64)
    low = load_dcm(low_data)
    low_img = low.pixel_array.astype(np.float64)
    if high.ImagerPixelSpacing[0] != high.ImagerPixelSpacing[1]:
        raise RuntimeError("Anisotropic pixels not supported")
    if low.ImagerPixelSpacing[0] != low.ImagerPixelSpacing[1]:
        raise RuntimeError("Anisotropic pixels not supported")
    if low.ImagerPixelSpacing[0] != high.ImagerPixelSpacing[0]:
        raise RuntimeError("Low and high images must have the same pixel spacing")
    if low_img.shape != high_img.shape:
        raise RuntimeError("Low and high images must have the same shape")
    yield high.ImagerPixelSpacing
    mask_inner_fraction = 0.8
    mask_outer_fraction = 1.2
    mask_outer_fraction2 = 1.44

    height_width = ROI / np.array(high.ImagerPixelSpacing)
    starts = ((np.array(high_img.shape) - height_width) // 2).astype(int)
    stops = np.ceil(starts + height_width).astype(int)
    slices = tuple((slice(s, e) for s, e in zip(starts, stops)))
    high_img = high_img[slices]
    low_img = low_img[slices]
    hough_centers = hough_wrapper(high_img, high.ImagerPixelSpacing[0])
    high_dt_img = quad_detrend(high_img, high.ImagerPixelSpacing[0], hough_centers)
    if quad_detrend_all:
        high_img = high_dt_img
        low_img = quad_detrend(low_img, low.ImagerPixelSpacing[0], hough_centers)
    yield high_dt_img, hough_centers
    hough_centers = sort_circles(high_dt_img, high.ImagerPixelSpacing[0], hough_centers)
    yield high_img, low_img, hough_centers
    trphantom, trradius = transform_phantom(hough_centers, read_phantom(), high.ImagerPixelSpacing[0])
    yield trphantom, trradius

    rad1 = trradius * mask_inner_fraction
    rad2 = trradius * mask_outer_fraction
    rad3 = trradius * mask_outer_fraction2

    yield rad1, rad2, rad3

    inner_labels, outer_labels = get_labels(high_img.shape, trphantom, rad1, rad2, rad3)
    yield inner_labels, outer_labels

    assert (trphantom[PMMA_MID_INDEX, 2] == 6.0)
    assert (trphantom[PMMA_MID_INDEX, 3] == 1.0)
    assert (trphantom[AL_MID_INDEX, 2] == 1.5)
    assert (trphantom[AL_MID_INDEX, 3] == 0.0)
    w_al = get_w(high_img, low_img,
                 inner_labels == PMMA_MID_INDEX + 1,
                 outer_labels == PMMA_MID_INDEX + 1)
    alimg = high_img / low_img ** w_al
    w_pmma = get_w(high_img, low_img,
                   inner_labels == AL_MID_INDEX + 1,
                   outer_labels == AL_MID_INDEX + 1)
    pmmaimg = high_img / low_img ** w_pmma
    yield w_pmma, w_al
    yield pmmaimg, alimg

    assert (np.all(trphantom[:, 2] == np.array([0.5, 1.0, 1.5, 2.0, 2.5, 10.0, 8.0, 6.0, 4.0, 2.0])))
    assert (np.all(trphantom[:, 3] == np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])))
    feature_inds = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
    feature_mats = ["Al"] * 5 + ["PMMA"] * 5
    yield feature_inds, feature_mats

    alimg_cnr_data = cnr(alimg, inner_labels, outer_labels, trphantom.shape[0],
                         feature_inds, feature_mats, trphantom[:, 2], "Al", air_kerma)
    pmmaimg_cnr_data = cnr(pmmaimg, inner_labels, outer_labels, trphantom.shape[0],
                           feature_inds, feature_mats, trphantom[:, 2], "PMMA", air_kerma)
    cnr_data = pd.DataFrame()
    cnr_data = cnr_data.append(alimg_cnr_data)
    cnr_data = cnr_data.append(pmmaimg_cnr_data)
    yield cnr_data


def proc(high_data, low_data, air_kerma, quad_detrend_all):
    prociter = _proc(high_data, low_data, air_kerma, quad_detrend_all)
    out = {}
    out["pixel_spacing"] = next(prociter)
    next(prociter)
    out["high_img"], out["low_img"], out["hough_centers"] = next(prociter)
    out["trphantom"], out["trradious"] = next(prociter)
    next(prociter)
    out["inner_labels"], out["outer_labels"] = next(prociter)
    out["w_pmma"], out["w_al"] = next(prociter)
    out["pmmaimg"], out["alimg"] = next(prociter)
    out["feature_inds"], out["feature_mats"] = next(prociter)
    cnr_data = next(prociter)
    try:
        next(prociter)
    except StopIteration:
        pass
    else:
        raise RuntimeError("Unexpected number of outputs")
    return cnr_data, out
