# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pd

dataset = pd.read_csv('Satisfaction_rate.csv')
# ////////////////////////////// Start Preprocessing ///////////////////////////
# ////////////////// Cleand Age ///////////////////

cleand_Age = []
default_Age = list(dataset.loc[:, 'age'].mode())[0]
for m in dataset.loc[:, 'age']:
    if str(m) == 'nan':
        cleand_Age.append(int(default_Age))
    else:
        cleand_Age.append(int(m))

# print(cleand_Age)

# ////////////////// Cleand Location ///////////////////

cleand_Location = []
default_Location = list(dataset.loc[:, 'location'].mode())[0]
for m in dataset.loc[:, 'location']:
    if str(m) == 'nan':
        cleand_Location.append(default_Location)
    else:
        cleand_Location.append(str(m))

# print(cleand_Location)
#
# # ////////////////// Cleand Education ///////////////////

cleand_Education = []
default_Education = list(dataset.loc[:, 'education'].mode())[0]
for m in dataset.loc[:, 'education']:
    if str(m) == 'nan':
        cleand_Education.append(default_Education)
    else:
        cleand_Education.append(str(m))

# print(cleand_Education)
#
# # ////////////////// Cleand Rating ///////////////////

cleand_Rating = []
default_Rating = dataset.loc[:, 'rating'].mean()
for m in dataset.loc[:, 'rating']:
    if str(m) == 'nan':
        cleand_Rating.append(int(default_Rating))
    else:
        cleand_Rating.append(int(m))

# print(cleand_Rating)
#
# # ////////////////// Cleand Awards ///////////////////

cleand_Awards = []
default_Awards = list(dataset.loc[:, 'awards'].mode())[0]
for m in dataset.loc[:, 'awards']:
    if str(m) == 'nan':
        cleand_Awards.append(int(default_Awards))
    else:
        cleand_Awards.append(int(m))

# print(cleand_Awards)

# ////////////////// Cleand Salary ///////////////////

cleand_Salary = []
default_Salary = dataset.loc[:, 'salary'].mean()
for m in dataset.loc[:, 'salary']:
    if str(m) == 'nan' or int(m) < 10000:
        cleand_Salary.append(int(default_Salary))
    else:
        cleand_Salary.append(int(m))

# print(cleand_Salary)

emp_id = dataset.loc[:, 'emp_id']
dept = dataset.loc[:, 'Dept']
recruitment_type = dataset.loc[:, 'recruitment_type']
job_level = dataset.loc[:, 'job_level']
onsite = dataset.loc[:, 'onsite']
certifications = dataset.loc[:, 'certifications']
satisfied = dataset.loc[:, 'satisfied']

cleand_data = pd.DataFrame({'id': emp_id, 'age': cleand_Age, 'dept': dept, 'location': cleand_Location,
                            'education': cleand_Education, 'recruitment_type': recruitment_type, 'job_level': job_level,
                            'rating': cleand_Rating, 'onsite': onsite, 'awards': cleand_Awards, 'certifications': certifications,
                            'salary': cleand_Salary, 'satisfied': satisfied})
cleand_data.to_csv('Satisfaction_cleand_rate.csv')
print(cleand_data)

# ////////////////////////////// End Preprocessing //////////////////////////