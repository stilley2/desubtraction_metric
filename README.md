# Introduction

This repository is an implementation of proposed IEC 62220-2 to assess image quality of dual energy x-ray imagers.

Nothing here will make sense if you are not familiar with the (as yet unpublished) standard. Sorry about that.

# Usage

There are currently two ways to use the repository, as a python library or as a [streamlit](https://streamlit.io/) app.

## Library

### Installation

1. `git clone https://github.com/stilley2/desubtraction_metric`
2. `cd desubtraction_metric`
3. `pip install .`

### Usage

```python
import desub

data, extra = desub.proc(high_energy_dicom_filename,  # file name of dicom file containing high energy data
                         low_energy_dicom_filename,  # file name of dicom file containing low energy data
                         air_kerma,  # air kerma in µGy
                         quad_detrend_all=False)  # whether to quadratically detrend before calculating the metric
```
where `data` is a [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
containing the output CNR for each DE image and feature and `extra` is a dictionary with intermediate processing
data.

## Streamlit app

### Heroku hosted

This app is hosted using heroku [here](https://desubtraction.herokuapp.com/).

### Self host

Self hosting is much faster.

1. `git clone https://github.com/stilley2/desubtraction_metric`
2. `pip install -r requirements.txt`
3. `streamlit run desub_app/app.py`
