import pandas as pd
import scipy.stats as stats

# Load the data from CSV
data = pd.read_csv("LungCancer25.csv")

# Clean up column names by removing leading/trailing whitespace
data.columns = data.columns.str.strip()

# Create a new DataFrame with the necessary columns
df = data[["Survival years", "Sex", "Surgery status"]].copy()


# Adjust surgery status and sex values
df.loc[:, "Surgery status"] = df["Surgery status"].replace({1: "No Surgery", 0: "Surgery"})
df.loc[:, "Sex"] = df["Sex"].replace({1: "Male", 2: "Female"})

# Separate data by sex and surgery status
male_surgery = df[(df["Sex"] == "Male") & (df["Surgery status"] == "Surgery")]["Survival years"].tolist()
male_no_surgery = df[(df["Sex"] == "Male") & (df["Surgery status"] == "No Surgery")]["Survival years"].tolist()
female_surgery = df[(df["Sex"] == "Female") & (df["Surgery status"] == "Surgery")]["Survival years"].tolist()
female_no_surgery = df[(df["Sex"] == "Female") & (df["Surgery status"] == "No Surgery")]["Survival years"].tolist()

# Perform statistical tests for male and female groups with surgery
male_surgery_mean = pd.Series(male_surgery).mean()
female_surgery_mean = pd.Series(female_surgery).mean()

print("Mean survival time for males with surgery:", male_surgery_mean)
print("Mean survival time for females with surgery:", female_surgery_mean)

# Perform statistical tests for male and female groups without surgery
male_no_surgery_mean = pd.Series(male_no_surgery).mean()
female_no_surgery_mean = pd.Series(female_no_surgery).mean()

print("Mean survival time for males without surgery:", male_no_surgery_mean)
print("Mean survival time for females without surgery:", female_no_surgery_mean)

# Perform Mann-Whitney U test for males with and without surgery
mwu_male = stats.mannwhitneyu(male_surgery, male_no_surgery, alternative='two-sided')
p_value_male = mwu_male.pvalue

# Perform Mann-Whitney U test for females with and without surgery
mwu_female = stats.mannwhitneyu(female_surgery, female_no_surgery, alternative='two-sided')
p_value_female = mwu_female.pvalue

# Print the p-values
print("Mann-Whitney U test p-value for males:", p_value_male)
print("Mann-Whitney U test p-value for females:", p_value_female)
