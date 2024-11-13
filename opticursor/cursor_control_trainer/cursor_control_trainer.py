# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib  # For saving and loading the scaler

# class CursorControlTrainer:
#     def __init__(self, data_file, model_save_path="saved_model/cursor_control_model.keras", scaler_save_path="saved_model/scaler.pkl"):
#         self.data_file = data_file
#         self.model_save_path = model_save_path
#         self.scaler_save_path = scaler_save_path
#         self.scaler = StandardScaler()
#         self.model = None

#     def load_and_preprocess_data(self):
#         # Load and preprocess data
#         data = pd.read_csv(self.data_file)
        
#         # X should only include the features (everything except 'mouse_x' and 'mouse_y')
#         X = data.drop(columns=['mouse_x', 'mouse_y']).values
        
#         # Y should only include the target labels (the actual 'mouse_x' and 'mouse_y' positions)
#         y = data[['mouse_x', 'mouse_y']].values
        
#         # Normalize the target values between 0 and 1 (if not already normalized)
#         y = np.clip(y, 0, 1)  # Ensure y values are between 0 and 1
        
#         # Fit and transform the features using the scaler (for X)
#         X = self.scaler.fit_transform(X)
        
#         # Split data into training and testing sets
#         return train_test_split(X, y, test_size=0.2, random_state=42)

#     def build_model(self, input_shape):
#         # Build the neural network model
#         self.model = Sequential([
#             Dense(128, activation='relu', input_shape=(input_shape,)),
#             Dropout(0.2),
#             Dense(64, activation='relu'),
#             Dense(32, activation='relu'),
#             Dense(2, activation='sigmoid')  # Apply sigmoid to ensure output is between 0 and 1
#         ])
#         self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

#     def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
#         # Train the model
#         self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
#         # Save the trained model
#         self.model.save(self.model_save_path)
#         print(f"Model saved to {self.model_save_path}")

#         # Save the fitted scaler
#         joblib.dump(self.scaler, self.scaler_save_path)
#         print(f"Scaler saved to {self.scaler_save_path}")

#     def run_training(self):
#         # Load and preprocess data, then train the model
#         X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
#         self.build_model(X_train.shape[1])
#         self.train_model(X_train, y_train, X_test, y_test)

# # Usage
# if __name__ == "__main__":
#     trainer = CursorControlTrainer("data/training_data.csv")
#     trainer.run_training()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving and loading the scaler

class CursorControlTrainer:
    def __init__(self, data_file, model_save_path="saved_model/cursor_control_model.keras", scaler_save_path="saved_model/scaler.pkl"):
        self.data_file = data_file
        self.model_save_path = model_save_path
        self.scaler_save_path = scaler_save_path
        self.scaler = StandardScaler()
        self.model = None

    def load_and_preprocess_data(self):
        # Load and preprocess data
        data = pd.read_csv(self.data_file)
        
        # X should only include the features (everything except 'mouse_x' and 'mouse_y')
        X = data.drop(columns=['mouse_x', 'mouse_y']).values
        
        # Y should only include the target labels (the actual 'mouse_x' and 'mouse_y' positions)
        y = data[['mouse_x', 'mouse_y']].values
        
        # Normalize the target values between 0 and 1 (if not already normalized)
        y = np.clip(y, 0, 1)  # Ensure y values are between 0 and 1
        
        # Fit and transform the features using the scaler (for X)
        X = self.scaler.fit_transform(X)
        
        # Reshape X to handle sequential data (if applicable, e.g., eye tracking over time)
        X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshaping into 3D for RNN input
        
        # Split data into training and testing sets
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self, input_shape):
        # Build a more complex model suited for sequential or spatial data
        self.model = Sequential([
            # Bidirectional LSTM layer to process spatial-temporal data (if applicable)
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(2, activation='sigmoid')  # Output layer: Predict mouse_x, mouse_y between 0 and 1
        ])
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        # Add a checkpoint to save the best model based on validation loss
        checkpoint = ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        
        # Train the model
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
        
        # Save the scaler
        joblib.dump(self.scaler, self.scaler_save_path)
        print(f"Scaler saved to {self.scaler_save_path}")

    def run_training(self):
        # Load and preprocess data, then train the model
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        self.build_model(X_train.shape[1:])  # Input shape corresponds to (timesteps, features)
        self.train_model(X_train, y_train, X_test, y_test)

# Usage example
if __name__ == "__main__":
    trainer = CursorControlTrainer("data/training_data.csv")
    trainer.run_training()
