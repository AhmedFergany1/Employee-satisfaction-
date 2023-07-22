from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



dataset = pd.read_csv('Satisfaction_cleand_rate.csv')

# x_train, x_test, y_train, y_test = train_test_split(
#     df_train.iloc[:, :], dataset.iloc[:, 12], test_size=0.1
# )

x_train, x_test, y_train, y_test = train_test_split(
   dataset.iloc[:, 2:13], dataset.iloc[:, 13], test_size=0.1
)

kmeans = KMeans(n_clusters=2)

kmeans.fit(x_train)

print(kmeans.cluster_centers_)

print(kmeans.labels_)

prediction = kmeans.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'prediction': prediction})
print(df)



