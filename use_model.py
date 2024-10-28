from ucimlrepo import fetch_ucirepo 

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
  # fetch dataset 
  bike_sharing = fetch_ucirepo(id=275) 

  # data (as pandas dataframes) 
  X = bike_sharing.data.features 
  y = bike_sharing.data.targets 

  # convert y to shape of (samples, )
  y = np.array(object=y)
  y = y.ravel()

  X = X.drop(["dteday", "yr"], axis=1)
  X = np.array(object=X)
  
  model = tf.keras.models.load_model('model.keras')

  print(model.summary())

  score = model.evaluate(X, y)

  print(f"RMS score: {np.sqrt(score)}")