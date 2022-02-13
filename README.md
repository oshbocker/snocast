# Snocast
This repo contains code to download data and train a model to predict snow water equivalent (SWE) values for 1 km^2 grids in the Western United States. It also contains code to use the trained model to make near real-time predictions for SWE.

This code is designed to run using Google Colab with Google Drive as a backend for storing data. The Google Colab notebooks in this repo assume that the directory structure below stems from a root of `/content/drive/MyDrive/snocast` and files are accessed from a Google Colab notebook after the google drive has been mounted.

```python
from google.colab import drive
drive.mount('/content/drive')
```


## Directory Structure
```
├── README.md                     <- The top-level README for developers using this project.
├── requirements.txt              <- The requirements file for reproducing the analysis environment, e.g.
│
├── train                         <- Code to acquire train data and train snocast model
│   ├── get_water_bodies_train_test.ipynb
│   ├── get_elevation_train_test.ipynb
│   ├── get_elevation_gradient_all.ipynb
│   ├── get_lccs_train_test.ipynb
│   ├── get_lccs_gm.ipynb
│   ├── get_modis_all.ipynb
│   ├── get_climate_all.ipynb
│   ├── train_model.ipynb
│   │
│   ├── data                      <- Scripts to download or generate data
│   │    ├── hrrr                 <- NOAA HRRR Climate data
│   │    ├── modis                <- Modis Terra and Aqua snow cover data outputs
│   │    └── static               <- Data sources that don't have a time element
│   │         ├── ground_measures_train_features.csv
│   │         ├── ground_measures_test_features.csv
│   │         ├── ground_measures_metadata.csv
│   │         ├── train_labels.csv
│   │         ├── labels_2020_2021.csv
│   │         ├── submission_format.csv
│   │         └── grid_cells.geojson
│   └──  models                   <- Model outputs
│
├── eval                          <- Code to acquire near real-time data and run predictions for snocast model
│   ├── get_water_bodies_eval.ipynb
│   ├── get_elevation_eval.ipynb
│   ├── get_lccs_eval.ipynb
│   ├── get_ground_measures_eval.ipynb
│   ├── get_modis_eval.ipynb
│   ├── get_climate_eval.ipynb
│   ├── snocast_model_predict.ipynb
│   │
│   ├── data                      <- Scripts to download or generate data
│   │   ├── ground_measures       <- Weekly Ground Measurments from SNOTEL and CDEC
│   │   ├── hrrr                  <- NOAA HRRR Climate data
│   │   ├── modis                 <- Modis Terra and Aqua snow cover data outputs
│   │   └── static                <- Data sources that don't have a time element
│   ├──  models                   <- Trained model outputs
│   └──  submissions              <- Submission outputs
```

## Train Instructions
In order to train the SWE prediction model a large quantity of historical data must be pulled into the directory structure above and then processed for the model. We assume that the following files will be manually downloaded from the [DrivenData website](https://www.drivendata.org/competitions/86/competition-reclamation-snow-water-eval/data/) and placed in the `train/data/static` directory.
* `ground_measures_train_features.csv`
* `ground_measures_test_features.csv`
* `ground_measures_metadata.csv`
* `train_labels.csv`
* `labels_2020_2021.csv`
* `submission_format.csv`
* `grid_cells.geojson`

With the base files - listed above - in place we set about acquiring the data necessary to train the SWE prediction model. For the sake of prediction the data are separated into two main categories, static and time-sensitive. Static data sources do not vary between prediction windows and represent geographical features of the grid cell. The time-varying data sources capture SWE features that will vary for a particular grid cell throughout the snow season. The three time-varying data sources are:
* Modis
* NOAA HRRR Climate Data
* Ground Measurements 

We will start by pulling the static data sources.
Note: when training the data all the sources are technically static since we are looking at historical measurements.

### Acquire Static Data for Train
Run the following notebooks in the `train` directory in any particular order. Some of the notebooks will require an AWS access key and secret, noted below.
* `get_water_bodies_train_test.ipynb` (requires AWS access key)
  * **Outputs:** 
    * `data/static/train_water.parquet`
    * `data/static/test_water.parquet`
* `get_lccs_train_test.ipynb` (requires AWS access key)
  * **Outputs:** 
    * `data/static/train_lccs.parquet`
    * `data/static/test_lccs.parquet`
* `get_lccs_gm.ipynb` (requires AWS access key)
  * **Outputs:** 
    * `data/static/train_gm.parquet`
* `get_elevation_train_test.ipynb`
  * **Outputs:** 
    * `data/static/train_elevation.parquet`
    * `data/static/test_elevation.parquet`
* `get_elevation_gradient_all.ipynb`
  * **Outputs:** 
    * `data/static/train_elevation_grads.parquet`
    * `data/static/test_elevation_grads.parquet`

### Acquire Modis Data for Train
Run the `get_modis_all.ipynb` notebook. This notebook will require access to [Google Earth Engine](https://developers.google.com/earth-engine). Since this notebook pulls data for each date in the train, test, and gm datasets, and the 15 days prior to each date, this notebook takes a very long time to run. Occassionally an error on the Colab Server or with the Google Earth Engine API will cause the program to quit. It is recommended to run the Colab notebook with Background Execution enabled and a High-RAM runtime.

<img width="445" alt="image" src="https://user-images.githubusercontent.com/1091020/153730313-43d3a41e-8374-464a-9a58-90328d5c595c.png">

**Outputs:** 
* Ground Measure Modis Data
  * `train/data/modis/modis_terra_gm.parquet`
  * `train/data/modis/modis_aqua_gm.parquet`
* Train Modis Data
  * `train/data/modis/modis_terra_train.parquet`
  * `train/data/modis/modis_aqua_train.parquet`
* Test Modis Data
  * `train/data/modis/modis_terra_test.parquet`
  * `train/data/modis/modis_aqua_test.parquet`

### Acquire NOAA HRRR Climate Data for Train
Run the `get_climate_all.ipynb` notebook. Since this notebook pulls data for each date in the train, test, and gm datasets, and the 3 days prior to each date, this notebook takes a very long time to run. Occassionally an error on the Colab Server or with the HRRR data storage locations will cause the program to quit. It is recommended to run the Colab notebook with Background Execution enabled and a High-RAM runtime.

**Outputs:** 
* Ground Measure Climate Data - `train/data/hrrr/gm_climate.parquet`
* Train Climate Data - `train/data/hrrr/train_climate.parquet`
* Test Climate Data - `test/data/hrrr/test_climate.parquet`

### Train the Model on the Acquired Data
Run the `train_model.ipynb` notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

This notebook will collate the data sources pulled in the previous steps, create relevant features for the model and finally train the model for SWE prediction.

**Outputs:** 
* Standard Scaler fit on the train data - `eval/models/std_scaler.bin`
  * Used to ensure that data scaling is similar when the model is run for prediction.
* XGBoost Model fit on the train data - `eval/models/xgb_all.txt`
  * Used as part of prediction ensemble.
* LGB Model fit on train data - `eval/models/lgb_all.txt`
  * Used as part of prediction ensemble. 

## Prediction (Eval) Instructions
First, run the `get_ground_measures_eval.ipynb` notebook which will pull the latest `ground_measures_features.csv` file from the `drivendata-public-assets` AWS S3 bucket and store it in the `eval/data/ground_measures` directory.

Next, open the `get_climate_eval.ipynb` notebook. Make sure to change the `run_date` variable to the current date for submitting predictions. `run_date` should be a string with `%Y-%m-%d` date format. Then run the notebook to generate the climate data which will be save to the `eval/data/hrrr` directory with the following format `climate_{run_date}.parquet`.
