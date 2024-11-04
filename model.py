from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class TimeToEventPredictor:
    def __init__(self, input_shape, lstm_units=128, checkpoint_dir='./checkpoints'):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            # First LSTM layer
            LSTM(units=self.lstm_units, 
                 return_sequences=True, 
                 input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(units=self.lstm_units, 
                 return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(units=self.lstm_units//2),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1)
        ])
        
        model.compile(optimizer='adam',
                     loss='huber',  # Huber loss is less sensitive to outliers
                     metrics=['mae'])
        
        model.summary()
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        # Create checkpoint callback
        checkpoint_path = self.checkpoint_dir / "model_best.h5"
        checkpoint = ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        
        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping],
            verbose=2
        )
        
        return history
    
    def load_best_model(self):
        """Load the best model from checkpoints"""
        best_model_path = self.checkpoint_dir / "model_best.h5"
        if best_model_path.exists():
            self.model = load_model(str(best_model_path))
            print("Loaded best model from checkpoints")
        else:
            print("No checkpoint found, using current model")
    
    def save_model(self, filename):
        """Save the current model to a specific file"""
        save_path = self.checkpoint_dir / filename
        self.model.save(str(save_path))
        print(f"Model saved to {save_path}")
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate_detailed(self, X_test, y_test):
        """Perform detailed evaluation with plots and metrics"""
        y_pred = self.predict(X_test)
        
        comparison = pd.DataFrame({
            'Actual_Hours': y_test,
            'Predicted_Hours': y_pred.flatten()
        })
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Create plots
        plt.figure(figsize=(8, 6))
        plt.scatter(comparison['Actual_Hours'], 
                   comparison['Predicted_Hours'], 
                   alpha=0.5)
        plt.xlabel('Actual Time Remaining (Hours)')
        plt.ylabel('Predicted Time Remaining (Hours)')
        plt.title('Actual vs. Predicted Time to Event')
        
        # Add diagonal line
        min_val = min(comparison.min().min(), 0)
        max_val = comparison.max().max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.show()
        
        # Print metrics
        print('\nDetailed Evaluation Metrics:')
        print(f'MAE: {mae:.2f} hours')
        print(f'RMSE: {rmse:.2f} hours')
        print(f'R² Score: {r2:.4f}')
        
        # Display prediction examples
        print('\nSample Predictions (in hours):')
        print(comparison.head())
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': comparison
        }