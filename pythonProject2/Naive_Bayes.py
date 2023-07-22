from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Satisfaction_cleand_rate.csv')

x_train, x_test, y_train, y_test = train_test_split(
    dataset.iloc[:, 2:13], dataset.iloc[:, 13], test_size=0.1
)

gnb = GaussianNB()

gnb.fit(x_train, y_train)

prediction = gnb.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'prediction': prediction})
print(df)

print(accuracy_score(y_test, prediction))