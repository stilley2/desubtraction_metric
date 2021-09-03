import io
import altair as alt
import streamlit as st
import matplotlib
from matplotlib.figure import Figure
from matplotlib import patches
import pandas as pd
import PIL.Image
import PIL.TiffImagePlugin


from desub import _proc


def to_tiff(img, pixel_spacing):
    x = PIL.Image.fromarray(img)
    ifd = PIL.TiffImagePlugin.ImageFileDirectory_v2()
    ifd[282] = 10.0 / pixel_spacing[1]  # x resolution
    ifd[283] = 10.0 / pixel_spacing[0]  # y resolution
    ifd[296] = 3  # centimeters
    fp = io.BytesIO()
    x.save(fp, format="tiff", tiffinfo=ifd)
    fp.seek(0)
    return fp


def plot_circles(img, circle_data, radii=None):
    mats = {0: "Al", 1: "PMMA"}
    fig_ = Figure()
    ax_ = fig_.add_subplot()
    ax_.imshow(img)
    for i in range(circle_data.shape[0]):
        ax_.plot(circle_data[i, 0], circle_data[i, 1], color='C0', marker='o', linestyle=None)
        if radii is not None:
            for radius in radii:
                ax_.add_patch(patches.Circle(circle_data[i, :2], radius=radius, facecolor='none', edgecolor='b'))
        if circle_data.shape[1] >= 4:
            ax_.text(circle_data[i, 0], circle_data[i, 1] - 5,
                     f"{circle_data[i, 2]:.2f} {mats[circle_data[i, 3]]}")
    return fig_


if __name__ == '__main__':
    st.set_page_config(page_title="DE Subtraction", page_icon="🔬")
    st.title("IEC 62220-2 Metric Evaluation")
    matplotlib.use("SVG")
    matplotlib.rcParams["image.cmap"] = "Greys_r"
    high_data_ = st.sidebar.file_uploader("High energy image file", type=["dcm", "DCM"])
    low_data_ = st.sidebar.file_uploader("Low energy image file", type=["dcm", "DCM"])
    quad_detrend_all_ = st.sidebar.checkbox("Quadratically detrend images prior to metric calculation")
    verbose = st.sidebar.checkbox("Verbose")
    air_kerma_ = st.sidebar.text_input("Air Kerma")
    if high_data_ is not None and low_data_ is not None and len(air_kerma_):
        air_kerma_ = float(air_kerma_)
        prociter = _proc(high_data_, low_data_, air_kerma_, quad_detrend_all_)
        pixel_spacing_ = next(prociter)
        high_dt_img_, hough_centers_ = next(prociter)
        if verbose:
            st.header("Quadratically detrended high image")
            st.pyplot(plot_circles(high_dt_img_, hough_centers_))
        high_img_, hough_centers_ = next(prociter)
        if verbose:
            st.header("High energy image with detected circle centers")
            st.pyplot(plot_circles(high_img_, hough_centers_))
        trphantom_, trradius_ = next(prociter)
        st.header("High energy image with registered ROIs")
        st.pyplot(plot_circles(high_img_, trphantom_, radii=[trradius_]))
        rad1_, rad2_, rad3_ = next(prociter)
        st.header("ROI mean/std extraction")
        st.pyplot(plot_circles(high_img_, trphantom_[:, :2], radii=[rad1_, rad2_, rad3_]))
        inner_labels_, outer_labels_ = next(prociter)
        if verbose:
            st.subheader("ROI Mask Visualization")
            st.text("Masks are shown by multipling the base image by 1.1")
            img_masks = high_img_.copy()
            img_masks[inner_labels_ > 0] *= 1.1
            img_masks[outer_labels_ > 0] *= 1.1
            fig = Figure()
            ax = fig.add_subplot()
            ax.imshow(img_masks)
            st.pyplot(fig)

        st.header("DE Images")
        w_pmma_, w_al_ = next(prociter)
        params = pd.DataFrame({"Material": ["Al", "PMMA"], "Subtraction parameter": [w_al_, w_pmma_]})
        st.dataframe(data=params)
        pmmaimg_, alimg_ = next(prociter)
        fig = Figure()
        ax = fig.add_subplot()
        ax.imshow(pmmaimg_)
        ax.set_title("PMMA subtracted image")
        pmmatiff = to_tiff(pmmaimg_, pixel_spacing_)
        st.download_button("Download PMMA subtracted image", pmmatiff, file_name="pmma.tiff", mime="image/tiff")
        st.pyplot(fig)
        fig = Figure()
        ax = fig.add_subplot()
        ax.imshow(alimg_)
        ax.set_title("Al subtracted image")
        altiff = to_tiff(alimg_, pixel_spacing_)
        st.download_button("Download Al subtracted image", altiff, file_name="al.tiff", mime="image/tiff")
        st.pyplot(fig)

        st.header("DE CNR")
        st.text("Feature indexes are separate for each material and ordered by increasing height")
        cnr_data_ = next(prociter)

        fig = alt.Chart(cnr_data_).mark_line(point=True)
        fig = fig.encode(x=alt.X("Feature Index", type="ordinal", axis=alt.Axis(labelAngle=0)),
                         y=alt.Y("CNR / √X", type="quantitative"),
                         color=alt.Color("Feature Material", type="nominal"),
                         row=alt.Row("DE Image", type="nominal", sort=["PMMA", "Al"]),
                         tooltip=["Feature Index", "CNR / √X",
                                  "Feature Material", "DE Image", "Feature Height (mm)"])
        fig = fig.configure_axis(labelFontSize=14, titleFontSize=16)
        fig = fig.configure_legend(labelFontSize=14, titleFontSize=16)
        fig = fig.configure_headerRow(labelFontSize=14, titleFontSize=16)
        st.altair_chart(fig, use_container_width=True)

        cnr_data_fp = io.StringIO()
        cnr_data_.to_csv(cnr_data_fp)
        cnr_data_fp.seek(0)
        st.download_button("Download CSV data", cnr_data_fp.read().encode("utf-8"),
                           file_name="desub.csv", mime="text/csv")

    st.markdown("Find source code on [github](https://github.com/stilley2/desubtraction_metric)")
    st.text("© 2021, Steven Tilley")
