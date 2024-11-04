# MIMIC-III Length of Stay Predictor
A deep learning system for predicting patient length of stay using sequential medical data.

## Slide 1: Data Loading and Preprocessing
### The Data Pipeline
- **Objective**: Load and prepare medical data for analysis.
- **Data Sources**:
  1. **CHARTEVENTS**: Contains vital signs like heart rate and blood pressure.
  2. **LABEVENTS**: Includes lab test results such as blood tests.
  3. **ADMISSIONS**: Details about hospital stays, including admission and discharge times.
  4. **PATIENTS**: Demographic information like age and gender.
- **Caching**: Saves processed data to avoid reloading large files every time, speeding up the process.
- **Column Handling**: Automatically recognizes column names in different cases (uppercase/lowercase) to ensure flexibility.

## Slide 2: Data Cleaning
### Making the Data Model-Ready
- **Timestamp Conversion**: Changes date and time strings into a format that Python can understand and manipulate.
- **Numeric Cleaning**: Removes non-numeric characters (like '%') from data to ensure all values are numbers.
- **Missing Values**: Uses techniques like forward and backward filling to fill in gaps in the data, ensuring no missing values disrupt the model.
- **Normalization**: Scales features to a range between 0 and 1 using MinMaxScaler, which helps the model learn more effectively.
- **Sequence Creation**: Groups data into sequences of 10 measurements per patient, allowing the model to learn from patterns over time.

## Slide 3: Target Variable
### Time Remaining Calculation
- **Purpose**: Calculate how many hours remain until a patient is discharged or passes away.
- **Calculation**: For each data point, compute the difference between the discharge time and the current time, converting it to hours.
  ```python
  hours_remaining = (discharge_time - current_time).total_seconds() / 3600
  ```
- **Normalization**: Adjusts the target variable (time remaining) to have a mean of 0 and a standard deviation of 1, making it easier for the model to learn.
  ```python
  y_normalized = (y - mean) / std_dev
  ```

## Slide 4: Model Architecture
### Deep LSTM Network
- **LSTM Layers**: Long Short-Term Memory (LSTM) layers are a type of neural network layer designed to handle sequences of data, like time series.
- **Batch Normalization**: Helps stabilize and speed up training by normalizing the output of each layer.
- **Dropout**: Randomly ignores some neurons during training to prevent overfitting, which is when a model learns the training data too well and performs poorly on new data.
- **Dense Layers**: Fully connected layers that help the model learn complex patterns.
- **Output Layer**: A single neuron that predicts the time remaining.

## Slide 5: Training Process
### Smart Training Strategy
- **Early Stopping**: Stops training when the model's performance on validation data stops improving, preventing overfitting.
- **Model Checkpointing**: Saves the best version of the model during training, ensuring you have the best model even if training is interrupted.
- **Adaptive Learning Rate**: Uses the Adam optimizer, which adjusts the learning rate during training for better performance.
- **Batch Size**: Processes data in batches of 64, balancing memory usage and training speed.
- **Validation Monitoring**: Uses validation loss to decide when to save the model and stop training.

## Slide 6: Evaluation
### Comprehensive Metrics
- **Mean Absolute Error (MAE)**: Measures the average error in hours between predicted and actual times.
- **Root Mean Square Error (RMSE)**: Similar to MAE but gives more weight to larger errors.
- **R² Score**: Indicates how well the model's predictions match the actual data, with 1 being a perfect match.
- **Visualization**: 
  - **Training/Validation Loss Curves**: Show how the model's error changes over time.
  - **Actual vs. Predicted Scatter Plots**: Visualize how close the model's predictions are to the actual values.

## Slide 7: Usage Example
### How to Use the Model
```python
# Initialize and load data
data_loader = MIMICDataLoader(data_dir='./data')
X, y_normalized, (y_mean, y_std) = data_loader.load_data()

# Create and train model
model = TimeToEventPredictor(input_shape=(10, n_features))
history = model.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = model.predict(X_test) * y_std + y_mean
```
- **Data Loading**: Prepares the data for training.
- **Model Training**: Trains the model on the data, adjusting weights to minimize error.
- **Prediction**: Uses the trained model to predict new data, converting normalized predictions back to actual hours.

## Slide 8: Key Features
1. **Robust Data Handling**: Efficiently processes large, complex datasets.
2. **Advanced Model Architecture**: Uses state-of-the-art techniques to handle sequential data.
3. **Production-Ready Features**: Includes features like checkpointing and early stopping for real-world applications.

## Slide 9: Future Improvements
1. **Feature Importance Analysis**: Determine which features most influence predictions.
2. **Uncertainty Quantification**: Measure the confidence of predictions.
3. **Interpretability Layers**: Make the model's decisions more understandable.
4. **Additional Metadata**: Incorporate more patient information for better predictions.
5. **Attention Mechanisms**: Use advanced techniques to focus on important parts of the data.

## Getting Started
1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Your Data Directory**:
   ```
   project/
   ├── data/
   │   ├── CHARTEVENTS.csv
   │   ├── LABEVENTS.csv
   │   ├── ADMISSIONS.csv
   │   └── PATIENTS.csv
   ```
3. **Run the Training**:
   ```bash
   python train.py
   ```