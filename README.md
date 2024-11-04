# Hospital Length of Stay Predictor
A deep learning system that predicts patient length of stay using MIMIC-III medical data.

## Project Overview
This project uses LSTM neural networks to predict how long a patient will remain in the hospital based on their medical data. It processes sequential medical data including vital signs and lab results to make these predictions.

## Features
- Processes MIMIC-III medical dataset
- Uses LSTM networks for time series prediction
- Includes data caching for faster processing
- Provides detailed visualizations of model performance
- Supports custom data paths and configurations

## Requirements
```bash
pandas
numpy
scikit-learn
tensorflow
pickle-mixin
matplotlib
```

## Project Structure
```
project/
├── data/                   # Data directory (not included)
│   ├── CHARTEVENTS.csv
│   ├── LABEVENTS.csv
│   ├── ADMISSIONS.csv
│   └── PATIENTS.csv
├── cache/                  # Cached processed data
├── checkpoints/            # Model checkpoints
├── data_loader.py         # Data processing module
├── model.py               # Neural network model
├── train.py               # Training script
└── requirements.txt       # Project dependencies
```

## Installation
1. Clone the repository
2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage
Run training with specified data path:
```bash
python train.py /path/to/mimic/data
```

Optional arguments:
```bash
python train.py /path/to/data --cache_dir ./my_cache --checkpoint_dir ./my_checkpoints
```

## Model Performance
The model achieves:
- MAE: ~0.2 hours on validation data
- Best performance for stays under 1500 hours
- Stable predictions for short to medium-term stays

### Performance Visualization
- Training metrics show consistent improvement
- Validation curves indicate some instability
- Prediction accuracy decreases for longer stays

## Model Architecture
```
LSTM Network:
- Input Layer
- LSTM(128) + BatchNorm + Dropout(0.3)
- LSTM(128) + BatchNorm + Dropout(0.3)
- LSTM(64) + BatchNorm + Dropout(0.3)
- Dense(64) + BatchNorm + Dropout(0.2)
- Dense(32) + BatchNorm + Dropout(0.2)
- Output Layer (1)
```

## Key Features
1. **Data Processing**:
   - Handles missing values
   - Normalizes features
   - Creates sequential data samples

2. **Model Training**:
   - Early stopping
   - Model checkpointing
   - Learning rate adaptation

3. **Evaluation**:
   - MAE and MSE metrics
   - Training/validation curves
   - Prediction visualization

## Limitations
- Decreased accuracy for stays over 1500 hours
- Some instability in validation performance
- Requires significant computational resources

## Future Improvements
1. Enhanced feature engineering
2. Separate models for different length stays
3. Additional regularization techniques
4. Uncertainty quantification
5. Attention mechanisms


