# Sales Horizon

An intuitive sales forecasting dashboard built with Python and Streamlit. Sales Horizon provides powerful time series forecasting capabilities with an easy-to-use interface for visualizing sales trends and predicting future performance.

## Features

- **Multiple Forecasting Methods**
  - Prophet (Facebook's advanced time series model)
  - Exponential Smoothing (Holt-Winters)
  - Moving Average
  - Naive Forecasting

- **Interactive Dashboard**
  - Real-time sales visualization with Plotly
  - Customizable forecast horizons (7-90 days)
  - 95% confidence intervals for predictions
  - Historical trend analysis

- **Comprehensive Analytics**
  - Sales distribution analysis
  - Weekly and monthly patterns
  - Forecast accuracy metrics (MAE, RMSE, MAPE)
  - Residual analysis with actual vs predicted scatter plots

- **Flexible Data Input**
  - Built-in sample data generation for testing
  - Upload your own CSV files
  - Filter by product category and region

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sales-horizon.git
cd sales-horizon
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run streamlit-app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Using Your Own Data

Your CSV file should have the following structure:
```csv
date,sales,product_category,region
2023-01-01,15000,Electronics,North
2023-01-02,18500,Electronics,South
2023-01-03,12000,Clothing,East
```

**Required columns:**
- `date`: Date in YYYY-MM-DD format
- `sales`: Numerical sales values

**Optional columns:**
- `product_category`: Product categories for filtering
- `region`: Geographic regions for filtering

### Configuration Options

#### Sidebar Controls

- **Data Source**: Choose between sample data or CSV upload
- **Forecast Horizon**: Select prediction period (7-90 days)
- **Forecasting Method**: Choose your preferred algorithm
- **Filters**: Filter data by category and region

#### Forecasting Methods

1. **Exponential Smoothing**: Best for data with trend and seasonality
2. **Moving Average**: Simple method for stable trends
3. **Naive Forecast**: Uses last known value as prediction
4. **Prophet**: Facebook's advanced model with automatic seasonality detection

## Dashboard Tabs

### Forecast
- Visual representation of historical and forecasted sales
- 95% confidence intervals
- Forecast summary table
- Key metrics (last actual, avg forecast, trend direction)

### Trends
- Sales distribution histogram
- Day-of-week analysis
- Monthly trend visualization

### Accuracy
- Forecast accuracy metrics (MAE, RMSE, MAPE)
- Actual vs Predicted scatter plot
- Residuals analysis over time

### Data
- View historical data
- Download data as CSV
- Data statistics summary
