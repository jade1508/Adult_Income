import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 2 Read the adult income dataset
df = pd.read_csv('adult_income_data.csv', header=None)

# Step 3 Create a script that will read a text file line by line
# Step 4 Add a name of Income for the response variable to the dataset
names = []
with open ("adult_income_names.txt") as f:
    for line in f:
        f.readline()
        var = line.split(":")[0]
        names.append(var)
names.append('income')

df.columns = names

# Step 5 Find the missing values
print(df.isnull().sum())

for c in df.columns:
    missing = df[c].isnull().sum()
    if missing > 0:
        print('{} has {} missing value(s)'.format(c,missing))
    else:
        print('{} has NO missing value!'.format(c))

# Step 6 Create a DataFrame with only age, education, and occupation by using subsetting
df_subset = df.loc [[i for i in range (32561)], ['age', 'education', 'occupation', 'native-country']]
print(df_subset)

# Step 7 Plot a histogram of age with a bin size of 20
plt.hist(df['age'], bins=20)
df_subset.boxplot(column='age', by='native-country', figsize=(15, 6))
plt.xticks(fontsize=5)
plt.xlabel('native-country', fontsize=15)
plt.show()

# Step 8 Create a function to strip the whitespace characters
def strip_whitespace (row): 
        return row.strip()

# Step 9 Use the apply method to apply this function to all the columns with string values,
df_subset['education_stripped'] = df['education'].apply(strip_whitespace)
df_subset['education'] = df_subset['education_stripped']
df_subset.drop(labels=['education_stripped'], axis=1, inplace=True)

# Step 10 Find the number of people who are aged between 30 and 50
df_filtered = df_subset[(df_subset['age'] >=30) &(df_subset['age'] <=50)]
print(df_filtered.head())

# Step 11 Group the records based on age and education to find how the mean age is distributed
print(df_subset.groupby(['native-country', 'education']).mean())

# Step 12 Group by occupation and show the summary statistics of age Find which profession has the oldest workers on average 
# and which profession has its largest share of the workforce above the 75 th percentile
print(df_subset.groupby('occupation').describe()['age'])

# Step 13 Use subset and groupby to find outliers
occupation_stats = df_subset.groupby('occupation').describe()['age']

# Step 14 Plot the values on a bar chart
plt.figure(figsize=(15,8))
plt.barh(y=occupation_stats.index,width=occupation_stats['count'])
plt.yticks(fontsize=13)
plt.show()

# Step 15 Merge the data using common keys
df1 = df[['age', 'workclass', 'occupation']].sample(5, random_state=101)
print(df1.head())

df2 = df[['education', 'native-country', 'occupation']].sample(5, random_state=101)
print(df2.head())

df_merged = pd.merge(df1, df2, on='occupation', how='inner').drop_duplicates()
print(df_merged)