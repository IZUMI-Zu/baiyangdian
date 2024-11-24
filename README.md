# Baiyangdian Water Level Prediction Based on Multi-Source Data Fusion

## Introduction

This project focuses on predicting water levels in Baiyangdian Lake using a multi-source data fusion approach. Baiyangdian is an important ecological and hydrological region, and accurate water level predictions are crucial for environmental conservation, flood management, and water resource planning. By integrating data from various sources, such as meteorological measurements, hydrological observations, and remote sensing, this model aims to improve prediction accuracy and reliability.

## Features

- **Multi-Source Data Integration**: Combines meteorological, hydrological, and remote sensing data for comprehensive analysis.
- **Advanced Prediction Models**: Utilizes machine learning algorithms, such as LSTM, GRU, or hybrid models, tailored for time-series water level forecasting.
- **Data Preprocessing**: Includes cleaning, normalization, and feature engineering steps to ensure high-quality input for the model.
- **Customizable Framework**: Allows users to adapt the model to other regions or datasets with minimal adjustments.
- **Visualization Tools**: Provides intuitive visualizations for prediction results and data trends.

## Data Sources

- **Meteorological Data**: Includes rainfall, temperature, and wind speed from local weather stations.
- **Hydrological Data**: Water levels and flow rates collected from observation stations in Baiyangdian.
- **Remote Sensing Data**: Satellite-based measurements, such as vegetation indices, water surface area, and soil moisture.
- **Other Environmental Data**: Optional datasets, such as groundwater levels or water quality metrics.

## Model Architecture

### Data Preprocessing Module

- Handles missing data, outliers, and feature extraction.
- Implements time-series alignment for multi-source datasets.

### Feature Fusion Module

- Combines structured and unstructured data through feature concatenation or attention mechanisms.
- Applies dimensionality reduction techniques if needed.

### Prediction Module

- Trains predictive models using algorithms like Long Short-Term Memory (LSTM) networks or Transformer-based models.
- Supports hyperparameter tuning for optimized performance.

### Evaluation Module

- Uses metrics such as RMSE, MAE, and R² to assess prediction accuracy.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/IZUMI-Zu/baiyangdian.git
2. Navigate to the project directory:

   ```bash
   cd baiyangdian
3. Install dependencies:

   ```bash
   poetry install
## Usage

1. Prepare the Data: Organize your datasets in the /data folder. Refer to the data_format.md for required formats.

2. Train the Model:

   ```bash
   python train.py --config config.yaml
3. Test the Model:

   ```bash
   python test.py --model checkpoint.pth --data test_data.csv
4. Visualize Results:

   ```bash
   python visualize.py --input predictions.csv
## Example Results

| **Model**        | **RMSE** | **MAE** | **R²** |
|-------------------|----------|----------|-------|
| Baseline (LSTM)   | 0.32 m   | 0.25 m   | 0.89  |
| Improved Model    | 0.28 m   | 0.21 m   | 0.92  |

## Future Work

- **Integrate Real-Time Data Streams**: Enable live predictions by incorporating real-time data streams from sensors and other sources.
- **Explore Deep Learning Models**: Investigate advanced models like graph neural networks (GNNs) for analyzing spatial-temporal data relationships.
- **Develop a Web-Based Dashboard**: Create an interactive dashboard for users to visualize and analyze prediction results in real time.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**: Create your own copy of the project.
2. **Create a New Branch**: Work on a new branch for your feature or bugfix.
3. **Submit a Pull Request**: Provide detailed descriptions of your changes when submitting a pull request.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.
