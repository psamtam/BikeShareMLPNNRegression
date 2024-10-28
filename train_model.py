from ucimlrepo import fetch_ucirepo 

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Normalization
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
  # fetch dataset 
  bike_sharing = fetch_ucirepo(id=275) 

  # data (as pandas dataframes) 
  X = bike_sharing.data.features 
  y = bike_sharing.data.targets 

  # convert y to shape of (samples, )
  y = np.array(object=y)
  y = y.ravel()

  # drop unnecessary features
  X = X.drop(["dteday", "yr"], axis=1)
  X = np.array(X)

  def my_model():
    model = Sequential()
    model.add(Input(X.shape[1]))
    model.add(Normalization())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error")
    print(model.summary())
    return model

  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

  print(X_train.shape, y_train.shape)
  print(X_test.shape, y_test.shape)

  model = my_model()

  history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)

  score = model.evaluate(X_train, y_train, verbose=0)
  print(f"Training score (MSE): {np.sqrt(score)}")
  score = model.evaluate(X_test, y_test, verbose=0)
  print(f"Testing score (MSE): {np.sqrt(score)}")

  model.save("model.keras")