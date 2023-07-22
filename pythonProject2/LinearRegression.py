from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


dataset = pd.read_csv('Satisfaction_cleand_rate.csv')

df_train = pd.DataFrame({'age': dataset.iloc[:, 2], 'dept': dataset.iloc[:, 3], 'location': dataset.iloc[:, 4],
                        'education': dataset.iloc[:, 5], 'recruitment_type': dataset.iloc[:, 6],
                        'job_level': dataset.iloc[:, 7], 'rating': dataset.iloc[:, 8], 'onsite': dataset.iloc[:, 9],
                        'awards': dataset.iloc[:, 10], 'certifications': dataset.iloc[:, 11],
                        'satisfied': dataset.iloc[:, 13]})

x_train, x_test, y_train, y_test = train_test_split(
    df_train.iloc[:, :], dataset.iloc[:, 12], test_size=0.1
)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
prediction = regressor.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'prediction': prediction})
print(df)

lin_mse = mean_squared_error(y_test, prediction)
lin_rmse = np.sqrt(lin_mse)
print()
print("Error ", lin_rmse)

# plt.plot(x_test, prediction, color="blue", linewidth=3)
# plt.show()

# print(accuracy_score(y_test, prediction))