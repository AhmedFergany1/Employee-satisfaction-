from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

dataset = pd.read_csv('Satisfaction_cleand_rate.csv')

# x = dataset.drop('satisfied', 1)
# y = dataset['satisfied']

x_train, x_test, y_train, y_test = train_test_split(
    dataset.iloc[:, 2:13], dataset.iloc[:, 13], test_size=0.1
)

supportVM = svm.SVC(kernel='rbf')
supportVM.fit(x_train, y_train)
# print(y_test)

prediction = supportVM.predict(x_test)
# print(prediction)
df = pd.DataFrame({'Actual': y_test, 'prediction': prediction})
print(df)

# lin_mse = mean_squared_error(y_test, prediction)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)
print()
print("Accuracy ",accuracy_score(y_test, prediction))