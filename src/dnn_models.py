import numpy as np
from scikeras.wrappers import KerasClassifier
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout


# DNN model 1 v0
# Paper: https://www.researchgate.net/profile/Madhusanka-Liyanage/publication/372250269_From_Opacity_to_Clarity_Leveraging_XAI_for_Robust_Network_Traffic_Classification/links/64acf0aac41fb852dd67fa41/From-Opacity-to-Clarity-Leveraging-XAI-for-Robust-Network-Traffic-Classification.pdf
# Source: https://github.com/Montimage/activity-classification/blob/master/xai/neural_networks_xai.ipynb
# Dataset: UNAC
class DNNClassifier1v0(KerasClassifier):
    def build(self, num_features, num_classes):
        # Creating a Keras Model
        model = Sequential()
        model.add(Dense(12, input_shape=(num_features,), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))
        # Compile the Keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # creating the model
        super().__init__(build_fn=model)

    def fit(self, X, y, **kwargs):
        # Add the desired parameters for fit
        kwargs['epochs'] = 150
        kwargs['batch_size'] = 10

        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        y_pred_one_hot = super().predict(X, **kwargs)
        y_pred = np.argmax(y_pred_one_hot, axis=1)
        return y_pred


# DNN model 1 v1
class DNNClassifier1v1(KerasClassifier):
    def build(self, num_features, num_classes):
        # Creating a Keras Model
        model = Sequential()
        model.add(Dense(128, input_shape=(num_features,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))
        # Compile the Keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # creating the model
        super().__init__(build_fn=model)

    def fit(self, X, y, **kwargs):
        # Add the desired parameters for fit
        kwargs['epochs'] = 150
        kwargs['batch_size'] = 10

        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        y_pred_one_hot = super().predict(X, **kwargs)
        y_pred = np.argmax(y_pred_one_hot, axis=1)
        return y_pred


# DNN model 2 v0
# Paper: //Rafael's code
# Source: https://github.com/rgtzths/XAI_analysis/blob/main/IOT_DNL/IOT_DNL.py
# Dataset: UNAC
class DNNClassifier2v0(KerasClassifier):
    def build(self, num_features, num_classes):
        # Creating a Keras Model
        model = Sequential()
        model.add(Flatten(input_shape=(num_features,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile the Keras model
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # creating the model
        super().__init__(build_fn=model)

    def fit(self, X, y, **kwargs):
        # Add the desired parameters for fit
        kwargs['epochs'] = 200
        kwargs['batch_size'] = 1024

        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        y_pred_one_hot = super().predict(X, **kwargs)
        y_pred = np.argmax(y_pred_one_hot, axis=1)
        return y_pred


# DNN model 2 v1
class DNNClassifier2v1(KerasClassifier):
    def build(self, num_features, num_classes):
        # Creating a Keras Model
        model = Sequential()
        model.add(Flatten(input_shape=(num_features,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.1))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.1))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile the Keras model
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # creating the model
        super().__init__(build_fn=model)

    def fit(self, X, y, **kwargs):
        # Add the desired parameters for fit
        kwargs['epochs'] = 200
        kwargs['batch_size'] = 1024

        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        y_pred_one_hot = super().predict(X, **kwargs)
        y_pred = np.argmax(y_pred_one_hot, axis=1)
        return y_pred