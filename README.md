# Snocast
This repo contains code to download data and train a model to predict snow water equivalent (SWE) values for 1 km^2 grids in the Western United States. It also contains code to use the trained model to make near real-time predictions for SWE.

This code is designed to run using Google Colab with Google Drive as a backend for storing data.


## Directory Structure
```
├── README.md                     <- The top-level README for developers using this project.
├── requirements.txt              <- The requirements file for reproducing the analysis environment, e.g.
│
├── train                         <- Code to acquire train data and train snocast model
│   ├── get_water_bodies.ipynb
│   ├── get_elevation.ipynb
│   │
│   ├── data                      <- Scripts to download or generate data
│   │    ├── hrrr                 <- NOAA HRRR Climate data
│   │    ├── modis                <- Modis Terra and Aqua snow cover data outputs
│   │    └── static               <- Data sources that don't have a time element
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

## Prediction (Eval) Instructions
First, run the `get_ground_measures_eval.ipynb` notebook which will pull the latest `ground_measures_features.csv` file from the `drivendata-public-assets` AWS S3 bucket and store it in the `eval/data/ground_measures` directory.

Next, open the `get_climate_eval.ipynb` notebook. Make sure to change the `run_date` variable to the current date to submit predictions for in `%Y-%m-%d` format. Then run the notebook to generate the climate data which will be save to the `eval/data/hrrr` directory with the following format `climate_{run_date}.parquet`.
