# -*- coding: utf-8 -*-
"""
Created on Tue May 23 00:18:20 2023

@author: ftuha
"""


# use the groupby function in pandas to calculate the counts of each combination
# of "Survival years" and "Sex"
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("LungCancer25.csv")

X = data [["Survival years", "Surgery status","Sex"]]
# Remove rows with Surgery status equal to 3
data = data[data["Surgery status"] != 3]


# Group the data by "Survival years" and "Sex" and count the occurrences
grouped_data = data.groupby(["Survival years", "Sex"]).size().unstack()

# Create the table with Survival years as the index
table = pd.DataFrame(index=range(1, 17), columns=["Survival years", "Total", "Male", "Female"])
table["Survival years"] = range(1, 17)
table["Total"] = grouped_data.sum(axis=1)
table["Male"] = grouped_data[1]
table["Female"] = grouped_data[2]

# Fill any missing values with 0
table = table.fillna(0)

# Print the table
print(table)