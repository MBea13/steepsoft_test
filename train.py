import numpy as np
import pickle
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping


class Train:
    def __init__(self, sleep_data, with_motion=True) -> None:
        self.__sequence_length = 3
        self.__sleep_data = sleep_data
        if with_motion:
            self.features = ['heart_rate', 'motion', 'step']
        else:
            self.features = ['heart_rate', 'step']
        self.create_train_test()

    @property
    def sequence_length(self):
        return self.__sequence_length

    @sequence_length.setter
    def sequence_length(self, val):
        self.__sequence_length = val

    def create_sequences(self):
        X, y = [], []
        for _, group in self.__sleep_data.groupby('subject_id'):
            group = group.reset_index(drop=True)
            for i in range(len(group) - self.__sequence_length):
                X.append(group.loc[i:i + self.__sequence_length - 1,
                         [feature for feature in self.features]].values)
                y.append(group.loc[i + self.__sequence_length, 'sleep_phase'])
        return np.array(X), np.array(y)

    def create_train_test(self):
        # self.scaler = StandardScaler()
        self.scaler = RobustScaler()
        # self.scaler = MinMaxScaler()
        self.__sleep_data[[f for f in self.features]] = self.scaler.fit_transform(
            self.__sleep_data[[f for f in self.features]])
        X, y = self.create_sequences()

        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=5 * 40, random_state=42)  # 5 samples per person for testing

    def build_model_with_best_param(self):
        if hasattr(self, "best_params"):
            print("Beast param found:", self.best_params)
        else:
            print("Calculating best params")
            self.find_best_parameter()
        self.build_model(self.best_params['units'])

    def build_model(self, units):
        self.model = Sequential()
        self.model.add(
            LSTM(
                units,
                input_shape=(
                    self.X_train.shape[1],
                    self.X_train.shape[2])))
        # Output layer with len units for the len(set) sleep phases
        self.model.add(
            Dense(len(set(self.__sleep_data.loc[:, 'sleep_phase'])), activation='softmax'))

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return self.model

    def train(self, units=64):
        self.build_model_with_best_param()
        # Define early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True)

        # Train the model with early stopping
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=100,
            batch_size=8,
            validation_split=0.1,
            callbacks=[early_stopping])
        # self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=8, validation_split=0.1)

    def evaluate(self):
        self.loss, self.accuracy = self.model.evaluate(
            self.X_test, self.y_test)
        print('Test Accuracy: {}\nLoss: {}'.format(self.accuracy, self.loss))

    def predict_sleep_phase(self, heart_rate, motion, steps):
        def dist(x, y): return np.linalg.norm(np.array(x) - np.array(y))
        if len(self.features) > 2:
            if hasattr(motion, '__len__'):
                motion = dist([0, 0, 0], motion)
            input_data = np.array([[heart_rate, motion, steps]])
        else:
            input_data = np.array([[heart_rate, steps]])
        input_data = self.scaler.fit_transform(
            input_data)  # Normalize the features

        # Reshape the input data to match the LSTM input shape
        # input_data = input_data.reshape((1, self.sequence_length, input_data.shape[1]))
        input_data = input_data.reshape(
            (input_data.shape[0], 1, input_data.shape[1]))

        # Repeat the input data to have a sequence length of 3
        input_data = np.repeat(input_data, 3, axis=1)

        # Predict sleep phase label
        predicted_label = self.model.predict(input_data, batch_size=8)

        # Decode the predicted label if necessary
        predicted_label = np.argmax(predicted_label, axis=1)

        return predicted_label[0]

    def dump(self, file_name):
        with open(sys.path[0] + "/" + file_name, 'wb') as f:
            pickle.dump(self, f)

    def find_best_parameter(self):
        # Create KerasClassifier wrapper
        model = KerasClassifier(build_fn=self.build_model)

        # Define hyperparameters to search over
        param_grid = {
            'units': [32, 64, 128],  # Number of LSTM units
        }

        # Perform grid search
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            verbose=1)
        grid_result = grid.fit(self.X_train, self.y_train)

        # Print results
        self.best_score = grid_result.best_score_
        self.best_params = grid_result.best_params_
        print(
            "Best: %f using %s" %
            (grid_result.best_score_,
             grid_result.best_params_))
