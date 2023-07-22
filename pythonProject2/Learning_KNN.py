from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


dataset = pd.read_csv('Satisfaction_cleand_rate.csv')

# dataset['satisfied'].replace(0, 'unsatisfied', inplace=True)
# dataset['satisfied'].replace(1, 'satisfied', inplace=True)

x_train, x_test, y_train, y_test = train_test_split(
    dataset.iloc[:, 2:13], dataset.iloc[:, 13], test_size=0.1
)
# print(y_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

prediction = knn.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'prediction': prediction})
print(df)
# print(prediction)

# lin_mse = mean_squared_error(y_test, prediction)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)
print()
print("Accuracy" , accuracy_score(y_test, prediction))

# dataset.drop('dept', axis=1, inplace=True)
# dataset.drop("location", axis=1, inplace=True)
# dataset.drop("education", axis=1, inplace=True)
# dataset.drop("recruitment_type", axis=1, inplace=True)

# age = dataset.loc[:, 'age']
# job_level = dataset.loc[:, 'job_level']
# cleand_Rating = dataset.loc[:, 'rating']
# onsite = dataset.loc[:, 'onsite']
# cleand_Awards = dataset.loc[:, 'awards']
# certifications = dataset.loc[:, 'certifications']
# cleand_Salary = dataset.loc[:, 'salary']
# satisfied = dataset.loc[:, 'satisfied']
#
# cleand_data = pd.DataFrame({'age': age, 'dept': z1, 'location': z2,
#                      'education': z3, 'recruitment_type': z4, 'job_level': job_level,
#                      'rating': cleand_Rating,'onsite': onsite, 'awards': cleand_Awards, 'certifications': certifications,
#                      'salary': cleand_Salary, 'satisfied': satisfied})



# dataset['age'].replace(27,'z',inplace=True)
