# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 07:50:11 2023

@author: ftuha
"""

import statsmodels.api as sm
import pandas as pd

import statsmodels.api as sm
import pandas as pd
import numpy as np

data = pd.read_csv("LungCancer25.csv")

X = data [["Sex", "Surgery status"]]

y = data['Survival years']

# add an intercept column to the dataframe
X['intercept'] = 1

# create interaction term for surgery and gender
X['surgery_gender'] = X['Surgery status'] * X['Sex']

# fit the Poisson regression model with main effects for surgery, gender, age, and interaction term
model = sm.GLM(y,X,
               family=sm.families.Poisson())
result = model.fit()

# print the model summary
print(result.summary())

# compute the adjusted mean survival time for males and females with and without surgery

# Filter the DataFrame to keep only the rows with 'male' gender

print("*********************************")
print("*********************************")

male_data = X[X['Sex'] == 1]
female_data = X[X['Sex'] == 2]

female_no_surger = female_data[female_data['Surgery status'] == 1]

female_surgery = female_data[female_data['Surgery status'] == 0]

male_no_surgery = male_data[male_data['Surgery status'] == 1]
male_surgery = male_data[male_data['Surgery status'] == 0]

mean_female_no_surgery = result.predict(female_no_surger)
mean_female_surgery = result.predict(female_surgery)
mean_male_no_surgery = result.predict(male_no_surgery)
mean_male_surgery = result.predict(male_surgery)

# print the adjusted mean survival time for males and females with and without surgery
print('Adjusted mean survival time for females with no surgery:', mean_female_no_surgery)
print('Adjusted mean survival time for females with surgery:', mean_female_surgery)
print('Adjusted mean survival time for males with no surgery:', mean_male_no_surgery)
print('Adjusted mean survival time for males with surgery:', mean_male_surgery)

print("*********************************")
print("*********************************")

male_data = X[X['Sex'] == 1]
femaledata = X[X['Sex'] == 2]

y_female = data[data['Sex'] == 2]
y_female = y_female['Survival years']

print("***************FEMALE***************")
print("***********************************")

print(femaledata.head())

# create interaction term for surgery and gender
femaledata['surgery_male'] = femaledata['Surgery status'] * femaledata['Sex']

model = sm.GLM(y_female,femaledata,
               family=sm.families.Poisson())
result = model.fit()

# print the model summary
print(result.summary())

p_value = result.pvalues['surgery_male']
if p_value < 0.05:
    print("The interaction between surgery and gender is significant (p-value={:.4f}).".format(p_value))
else:
    print("The interaction between surgery and gender is not significant (p-value={:.4f}).".format(p_value))


print("***********************************")
print("***********************************")


print("=================================")
print("=================================")


print(result.conf_int())
print("=================================")
print("=================================")
# Calculate hazard ratios (HR)
HR = pd.DataFrame({'HR': np.exp(result.params),
                   'p-value': result.pvalues})

print(HR)

print("=================================")
print("=================================")



