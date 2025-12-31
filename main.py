import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import shap as sh


data = pd.read_csv('receiving_week11.csv', index_col='Rk')
columnstodrop = ["Player", "Age", "Team", "Pos", "R/G"]
inputdata = data.drop(columns=columnstodrop, axis=1)
cleaninput = inputdata.dropna()
cleaninput = cleaninput.head(300)

x = cleaninput.drop(columns="Rec", axis=1)
y = cleaninput["Rec"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
sta = StandardScaler()
x_train = sta.fit_transform(x_train)
x_test = sta.transform(x_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=32, activation='relu'))

ann.add(tf.keras.layers.Dense(units=32, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation=None))

#for identification prediction
#ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#for regression prediction
ann.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = ann.fit(x_train, y_train, batch_size=32, epochs=200, validation_split=0.2)
y_pred = ann.predict(x_test)


# Plot accuracy
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Plot loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model Loss (MSE)')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Val'], loc='upper right')
#plt.show()


plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.show()



