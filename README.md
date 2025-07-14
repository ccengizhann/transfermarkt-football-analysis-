# Transfermarkt Football Data Analysis Project

This project is designed for comprehensive football data analysis using the [Transfermarkt API](https://transfermarkt-api.fly.dev/).

## 🎯 Project Objectives

- **Player Analysis**: Performance metrics, market value analysis
- **Transfer Analysis**: Transfer trends, price predictions
- **Team Performance**: League comparisons, team analysis
- **Machine Learning**: Market value prediction models

## 📁 Project Structure

```
transfermarkt-analysis/
├── data/                    # Raw and processed data
├── notebooks/              # Jupyter notebook analyses
├── src/                    # Main source codes
│   ├── data_collection/    # API data collection
│   ├── analysis/          # Data analysis functions
│   ├── visualization/     # Visualization tools
│   └── models/           # ML models
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Installation

### Requirements
- Python 3.8+
- pip or conda

### Installation Steps

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

3. **Open main analysis notebook:**
```bash
notebooks/01_data_exploration.ipynb
```

## 📊 Data Source

This project uses the [Transfermarkt API](https://transfermarkt-api.fly.dev/):

### Available Endpoints:
- **Players**: Profile, statistics, market value, transfers
- **Clubs**: Profile, player squads, search
- **Leagues**: Clubs, search
- **Transfers**: Player transfer history

### API Example Usage:
```python
import requests

# Player search
response = requests.get("https://transfermarkt-api.fly.dev/players/search/messi")

# Player market value
response = requests.get("https://transfermarkt-api.fly.dev/players/28003/market_value")
```

## 🔍 Analysis Areas

### 1. Player Analysis
- Market value trends
- Age vs performance correlation
- Position-based comparisons

### 2. Transfer Analysis
- Transfer price trends
- Most profitable transfers
- League-based transfer analysis

### 3. Team Performance
- Team value analysis
- Player age distributions
- League comparisons

### 4. Machine Learning
- Market value prediction models
- Transfer success prediction
- Player performance classification

## 🛠 Technologies Used

- **Data Collection**: `requests`, `pandas`
- **Data Analysis**: `pandas`, `numpy`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Development**: `jupyter`, `pytest`

## 📈 Outputs

Main outputs you will obtain from this project:
- Detailed data analysis reports
- Interactive visualizations
- Market value prediction model
- Transfer pattern analysis
- League comparison dashboard


