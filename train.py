from data_loader import MIMICDataLoader
from model import TimeToEventPredictor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train MSE')
    plt.plot(history.history['val_loss'], label='Validation MSE')
    plt.title('Model Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('Hours')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Initialize data loader
    data_loader = MIMICDataLoader(data_dir='./data')
    
    # Load and prepare data
    X, y_normalized, (y_mean, y_std) = data_loader.load_data(sequence_length=10)
    
    # Split the data without stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_normalized, test_size=0.2, random_state=42
    )
    
    # Initialize model with checkpoint directory
    model = TimeToEventPredictor(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        checkpoint_dir='./checkpoints'
    )
    
    # Train the model
    history = model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=64)
    
    # Plot training history
    plot_training_history(history)
    
    # Load the best model before evaluation
    model.load_best_model()
    
    # Make predictions and denormalize
    y_pred_normalized = model.predict(X_test)
    y_pred = y_pred_normalized * y_std + y_mean
    y_test_denorm = y_test * y_std + y_mean
    
    # Calculate metrics on denormalized values
    mae = mean_absolute_error(y_test_denorm, y_pred)
    mse = mean_squared_error(y_test_denorm, y_pred)
    rmse = np.sqrt(mse)
    
    print('\nModel Evaluation (in hours):')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_denorm, y_pred, alpha=0.5)
    plt.plot([y_test_denorm.min(), y_test_denorm.max()], 
             [y_test_denorm.min(), y_test_denorm.max()], 'r--')
    plt.xlabel('Actual Time Remaining (hours)')
    plt.ylabel('Predicted Time Remaining (hours)')
    plt.title('Actual vs Predicted Time Remaining')
    plt.show()

if __name__ == "__main__":
    main() 