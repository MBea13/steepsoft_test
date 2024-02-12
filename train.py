import numpy as np
import pickle
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.utils.class_weight import compute_class_weight


def delete_rows(df, ids, column, sc=0.75):
    df['consecutive_count'] = (df[column] != df[column].shift(1)).cumsum()
    counts = df.groupby([column, 'consecutive_count']).size().reset_index(name='count')
    mean = int(counts['count'].mean())
    m = int(1.5 * mean)
    groups = df.groupby([ids, column])
    temp_df = pd.DataFrame()
    for _, g in groups:
        sz = g.shape[0]
        if sz > (m * sc):
            gg = g.drop(g.index[int(m * sc) : sz - int(m * sc)])
        else:
            gg = g
        temp_df = pd.concat([temp_df, gg])
    return temp_df, m

def detect_outliers(df, features):
    Q1 = df[features].quantile(0.25)
    Q3 = df[features].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[features] < lower_bound) | (df[features] > upper_bound)
    return outliers


def get_average_sleep_phase_length(df, column):
# def average_length_consecutive_sequences(df, column):
    lengths = []
    current_length = 0
    current_value = None

    # Iterate over the DataFrame
    for value in df[column]:
        if value != current_value:
            if current_value is not None:
                lengths.append(current_length)
            current_length = 1
            current_value = value
        else:
            current_length += 1

    # Append the last sequence length
    lengths.append(current_length)

    # Calculate the average length
    if len(lengths) > 0:
        average_length = sum(lengths) / len(lengths)
    else:
        average_length = 0

    return average_length


class Train:
    def __init__(self, sleep_data, with_motion=True) -> None:
        self.__sleep_data = sleep_data
        # replacing 5s to 4s TODO ask what happened here
        self.__sleep_data.loc[self.__sleep_data["sleep_phase"] == 5, "sleep_phase"] = 4
        self.__sleep_data, _ = delete_rows(self.__sleep_data, 'subject_id', 'sleep_phase', sc = 1.2)
        if with_motion:
            self.features = ['heart_rate', 'motion', 'step']
        else:
            self.features = ['heart_rate', 'step']
        self.subject_nr = len(set(self.__sleep_data['subject_id']))
        self.__sequence_length = int(get_average_sleep_phase_length(self.__sleep_data, 'sleep_phase') * 1.2)
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []
        self.create_train_test()
        
        # # Calculate class weights
        self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train)
        # Convert class weights to a dictionary
        self.class_weight_dict = dict(enumerate(self.class_weights))
              
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = plt.subplot(111)
        self.__sleep_data['sleep_phase'].hist(ax=ax)
        fig.savefig('labels_hist.png')

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
            for i in range(len(group)):
                features = group.loc[i:i + self.__sequence_length - 1,
                         [feature for feature in self.features]].values
                padding_length = max(0, self.__sequence_length - len(features))
                X.append(np.pad(features, ((0, padding_length), (0, 0)), mode='edge'))
                y.append(group.loc[i, 'sleep_phase'])         
        return np.array(X), np.array(y)
        

    def create_train_test(self):
        outliers = detect_outliers(self.__sleep_data, self.features)
        self.__sleep_data = self.__sleep_data[~outliers.any(axis=1)]
        self.scaler = RobustScaler()
        self.__sleep_data[[f for f in self.features]] = self.scaler.fit_transform(
            self.__sleep_data[[f for f in self.features]])
        X, y = self.create_sequences()
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=5 * self.subject_nr, random_state=42)  # 5 samples per person for testing

    def build_model_with_best_param(self):
        """If the object was pickled previously we will use its elready calculated hyperparameters
        """
        if hasattr(self, "best_params"):
            print("Best param found:", self.best_params)
        else:
            print("Calculating best params")
            self.find_best_parameter()
        self.build_model(self.best_params['units'], self.best_params['dropout_rate'])
    
    def build_model(self, units, dropout_rate=0.2):
        # Build stacked LSTM model
        self.model = Sequential()
        self.model.add(LSTM(units, return_sequences=False, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))

        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(units, activation='relu'))
        self.model.add(Dense(len(set(self.__sleep_data.loc[:, 'sleep_phase'])), activation='softmax'))

        # Compile the model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model
    
    
    def train(self):
        self.build_model_with_best_param()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True)
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=70,
            batch_size=self.best_params['batch_size'],
            validation_split=0.1,
            callbacks=[early_stopping], class_weight=self.class_weight_dict)
        
        # For plot training and validation loss
        self.train_loss.append(history.history['loss'])
        self.val_loss.append(history.history['val_loss'])
        self.train_accuracy.append(history.history['accuracy'])
        self.val_accuracy.append(history.history['val_accuracy'])

    def evaluate(self):
        temp_dict = dict.fromkeys([val for val in self.y_test], 0)
        for x in temp_dict.keys():
            temp_dict[x] = (100 * list(self.y_test).count(x)) / len(self.y_test)
            
        print("Labels percentage")
        print(temp_dict)
        
        self.loss, self.accuracy = self.model.evaluate(
            self.X_test, self.y_test)
        
        print('Test Accuracy: {}\nLoss: {}'.format(self.accuracy, self.loss))
        
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print('Test Accuracy: {}\nLoss: {}'.format(accuracy, loss))
        
        
    def predict_sleep_phase(self, heart_rate, motion, steps):
    #     """
    #     Predicts the sleep phase based on given features.

    #     Args:
    #         heart_rate: A single value representing the heart rate.
    #         motion: A single value representing the motion.
    #         steps: A single value representing the number of steps.

    #     Returns:
    #         The predicted sleep phase as an integer.
    #     """

    #     # Preprocess the input data
    #     def dist(x, y): return np.linalg.norm(x - y)
    #     data = np.array([heart_rate, dist([0, 0, 0], motion), steps]).reshape(1, -1, len(self.features))


    #     # Pad the data if necessary
    #     if data.shape[1] < self.__sequence_length:
    #         padding_length = self.__sequence_length - data.shape[1]
    #         data = np.pad(data, ((0, 0), (0, padding_length), (0, 0)), mode='edge')

    #     # Make the prediction
    #     prediction = self.model.predict(data)[0]

    #     # Return the predicted sleep phase with the highest probability
    #     return np.argmax(prediction)
    
    
        # Preprocess the input data
        input_data = np.array([heart_rate, motion, steps]).reshape(1, -1)

        # Scale the input data using the same scaler used during training
        scaled_input_data = self.scaler.transform(input_data)

        # Reshape the data to match the LSTM input shape
        data = scaled_input_data.reshape(1, -1, len(self.features))
        # print("------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>zz ", data)

        # Pad the data if necessary
        if data.shape[1] < self.__sequence_length:
            padding_length = self.__sequence_length - data.shape[1]
            data = np.pad(data, ((0, 0), (0, padding_length), (0, 0)), mode='edge')

        # Make the prediction
        prediction = self.model.predict(data)[0]

        # Return the predicted sleep phase with the highest probability
        return np.argmax(prediction)
    

    def dump(self, file_name):
        with open(sys.path[0] + "/" + file_name, 'wb') as f:
            pickle.dump(self, f)

    def find_best_parameter(self):
        """Setting hyperparameters to try and optimalize the system
        """
        # Create KerasClassifier wrapper
        model = KerasClassifier(build_fn=self.build_model)

        # Define hyperparameters to search over
        param_grid = {
            'units': [32, 64, 128],  # Number of LSTM units
            'dropout_rate': [0.2, 0.3],  # Different dropout values
            'batch_size': [8, 16, 32],  # Different batch sizes to try
        }

        # Perform grid search
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            verbose=1)
        grid_result = grid.fit(self.X_train, self.y_train, class_weight=self.class_weight_dict)

        # Print results
        self.best_score = grid_result.best_score_
        self.best_params = grid_result.best_params_
        print(
            "Best: %f using %s" %
            (grid_result.best_score_,
             grid_result.best_params_))
