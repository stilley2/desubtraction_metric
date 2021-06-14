import altair as alt
import streamlit as st
import pandas as pd
import pydicom
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib import patches
import skimage
import skimage.feature
import skimage.transform
import csv


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

@st.cache
def read_phantom():
    with open("phantom_spec.txt", "r", newline='') as f:
        r = csv.reader(f)
        next(r)  # skip first line
        phantom = [rt for rt in r]
        phantom = np.array(phantom).astype(np.float64)
    return phantom


@st.cache
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


@st.cache
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


@st.cache
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


@st.cache
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


def plot_circles(img, circle_data, radii=None):
    mats = {0: "Al", 1: "PMMA"}
    fig = Figure()
    ax = fig.add_subplot()
    ax.imshow(img)
    for i in range(circle_data.shape[0]):
        ax.plot(circle_data[i, 0], circle_data[i, 1], color='C0', marker='o', linestyle=None)
        if radii is not None:
            for radius in radii:
                ax.add_patch(patches.Circle(circle_data[i, :2], radius=radius, facecolor='none', edgecolor='b'))
        if circle_data.shape[1] >= 4:
            ax.text(circle_data[i, 0], circle_data[i, 1] - 5,
                    f"{circle_data[i, 2]:.2f} {mats[circle_data[i, 3]]}")
    return fig


@st.cache
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


@st.cache
def get_w(hi, li, minner, mouter):
    #return np.log(np.mean(hi[mouter]) / np.mean(hi[minner])) / np.log(np.mean(li[mouter]) / np.mean(li[minner]))
    return np.log(np.mean(hi[minner]) / np.mean(hi[mouter])) / np.log(np.mean(li[minner]) / np.mean(li[mouter]))


@st.cache
def cnr(img, inner, outer, nlabels):
    cnr = []
    for i in range(nlabels):
        im = inner == i + 1
        om = outer == i + 1
        cnr.append(np.abs(np.mean(img[im]) - np.mean(img[om])) / np.sqrt(np.var(img[im]) + np.var(img[om])))
    return np.array(cnr)


def load_dcm(f):
    try:
        dataset = pydicom.dcmread(f)
    except (AttributeError, pydicom.errors.InvalidDicomError):
        f.seek(0)
        pydicom.filereader.read_preamble(f, False)
        h = pydicom.filereader.read_dataset(f, is_implicit_VR=False, is_little_endian=True, stop_when=lambda tag, vr, val: tag.group > 2)
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


if __name__ == '__main__':
    st.set_page_config(page_title="DE Subtraction", page_icon="🔬")
    st.title("IEC 62220-2 Metric Evaluation")
    matplotlib.use("SVG")
    matplotlib.rcParams["image.cmap"] = "Greys_r"
    high_data = st.sidebar.file_uploader("High energy image file", type=["dcm", "DCM"])
    low_data = st.sidebar.file_uploader("Low energy image file", type=["dcm", "DCM"])
    quad_detrend_all = st.sidebar.checkbox("Quadratically detrend images prior to metric calculation")
    verbose = st.sidebar.checkbox("Verbose")
    mask_inner_fraction = st.sidebar.text_input("Inner ROI radius fraction",
                                                value=0.8,
                                                help="Multiply the feature radius by this value to get the radius "
                                                     "of the inner mask.")
    mask_outer_fraction = st.sidebar.text_input("Outer ROI radius fraction",
                                                value=1.2,
                                                help="Multiply the feature radius by this value to get the inner "
                                                     "radius of the outer mask. The outer radius is chosen to "
                                                     "match the area of the inner mask.")
    air_kerma = st.sidebar.text_input("Air Kerma")
    mask_inner_fraction = float(mask_inner_fraction)
    mask_outer_fraction = float(mask_outer_fraction)
    if high_data is not None and low_data is not None and len(air_kerma):
        air_kerma = float(air_kerma)
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
        if verbose:
            st.header("Quadratically detrended high image")
            st.pyplot(plot_circles(high_dt_img, hough_centers))
        hough_centers = sort_circles(high_dt_img, high.ImagerPixelSpacing[0], hough_centers)
        if verbose:
            st.header("High energy image with detected circle centers")
            st.pyplot(plot_circles(high_img, hough_centers))
        trphantom, trradius = transform_phantom(hough_centers, read_phantom(), high.ImagerPixelSpacing[0])
        st.header("High energy image with registered ROIs")
        st.pyplot(plot_circles(high_img, trphantom, radii=[trradius]))

        rad1 = trradius * mask_inner_fraction
        rad2 = trradius * mask_outer_fraction
        rad3 = np.sqrt(rad1 ** 2 + rad2 ** 2)

        st.header("ROI mean/std extraction")
        st.pyplot(plot_circles(high_img, trphantom[:, :2], radii=[rad1, rad2, rad3]))
        inner_labels, outer_labels = get_labels(high_img.shape, trphantom, rad1, rad2, rad3)
        if verbose:
            st.subheader("ROI Mask Visualization")
            st.text("Masks are shown by multipling the base image by 1.1")
            img_masks = high_img.copy()
            img_masks[inner_labels > 0] *= 1.1
            img_masks[outer_labels > 0] *= 1.1
            fig = Figure()
            ax = fig.add_subplot()
            ax.imshow(img_masks)
            st.pyplot(fig)
        assert(trphantom[PMMA_MID_INDEX, 2] == 6.0)
        assert(trphantom[PMMA_MID_INDEX, 3] == 1.0)
        assert(trphantom[AL_MID_INDEX, 2] == 1.5)
        assert(trphantom[AL_MID_INDEX, 3] == 0.0)
        w_al = get_w(high_img, low_img,
                     inner_labels == PMMA_MID_INDEX + 1,
                     outer_labels == PMMA_MID_INDEX + 1)
        alimg = high_img / low_img ** w_al
        w_pmma = get_w(high_img, low_img,
                       inner_labels == AL_MID_INDEX + 1,
                       outer_labels == AL_MID_INDEX + 1)
        pmmaimg = high_img / low_img ** w_pmma

        st.header("DE Images")
        params = pd.DataFrame({"Material": ["Al", "PMMA"], "Subtraction parameter": [w_al, w_pmma]})
        st.dataframe(data=params)
        fig = Figure()
        ax = fig.add_subplot()
        ax.imshow(pmmaimg)
        ax.set_title("PMMA subtracted image")
        st.pyplot(fig)
        fig = Figure()
        ax = fig.add_subplot()
        ax.imshow(alimg)
        ax.set_title("Al subtracted image")
        st.pyplot(fig)

        st.header("DE CNR")
        st.text("Feature indexes are separate for each material and ordered by increasing height")
        assert(np.all(trphantom[:, 2] == np.array([0.5, 1.0, 1.5, 2.0, 2.5, 10.0, 8.0, 6.0, 4.0, 2.0])))
        assert(np.all(trphantom[:, 3] == np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])))
        feature_inds = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])

        alimg_cnr = cnr(alimg, inner_labels, outer_labels, trphantom.shape[0])
        pmmaimg_cnr = cnr(pmmaimg, inner_labels, outer_labels, trphantom.shape[0])
        alimg_cnr /= np.sqrt(air_kerma)
        pmmaimg_cnr /= np.sqrt(air_kerma)
        pmma_feature_inds = trphantom[:, 3] == 1
        al_feature_inds = trphantom[:, 3] == 0
        cnr_data = pd.DataFrame({"Feature Index": feature_inds,
                                 "PMMA Image CNR / √X": pmmaimg_cnr,
                                 "Al Image CNR / √X": alimg_cnr,
                                 "Feature Material": ["Al"] * 5 + ["PMMA"] * 5,
                                 "Feature Height (mm)": trphantom[:, 2]})

        fig = alt.Chart(cnr_data).mark_line(point=True)
        fig = fig.encode(x=alt.X("Feature Index", type="ordinal", axis=alt.Axis(labelAngle=0)),
                         y="PMMA Image CNR / √X", color="Feature Material",
                         tooltip=["Feature Index", "PMMA Image CNR / √X",
                                  "Feature Material", "Feature Height (mm)"])
        fig = fig.configure_axis(labelFontSize=14, titleFontSize=16).configure_legend(labelFontSize=14, titleFontSize=16)
        st.altair_chart(fig, use_container_width=True)

        fig = alt.Chart(cnr_data).mark_line(point=True)
        fig = fig.encode(x=alt.X("Feature Index", type="ordinal", axis=alt.Axis(labelAngle=0)),
                         y="Al Image CNR / √X", color="Feature Material",
                         tooltip = ["Feature Index", "PMMA Image CNR / √X",
                                    "Feature Material", "Feature Height (mm)"])
        fig = fig.configure_axis(labelFontSize=14, titleFontSize=16).configure_legend(labelFontSize=14, titleFontSize=16)
        st.altair_chart(fig, use_container_width=True)

    st.markdown("Find source code on [github](https://github.com/stilley2/desubtraction_metric)")
    st.text("© 2021, Steven Tilley")
