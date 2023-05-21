# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 02:33:56 2023

@author: ftuha
"""

from scipy import stats
import pandas as pd

# create the DataFrame with the given data
data = pd.read_csv("LungCancer25.csv")

# separate the survival times for male and female subjects
male_survival_times = data[data['Sex'] == 1]['Survival years']
female_survival_times = data[data['Sex'] == 2]['Survival years']

# perform a t-test for independent samples
t_statistic, p_value = stats.ttest_ind(male_survival_times, female_survival_times)

# print the result

print ("pvalue:", p_value)
print ("t statistic", t_statistic)
if p_value < 0.05:
    if t_statistic > 0:
        print("Male survival time is significantly lower than female survival time.")
    else:
        print("Female survival time is significantly lower than male survival time.")
else:
    print("There is no significant difference in survival time between male and female.")
