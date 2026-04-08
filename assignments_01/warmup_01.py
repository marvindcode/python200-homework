# %% [markdown]
# # Lesson 01: Analysis Intro

# %% [markdown]
# ## Part 1: Warmup Exercises

# %% [markdown]
# ### Pandas Question 1
# Create the following DataFrame and print the first three rows, the shape, and the data types of each column
# Print each result with a label (e.g. print(f"Num Rows: {len(df)}")).

# %%
import pandas as pd

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f"First three rows: {df.head(3)}"),
print(f"Shape of dataset: {df.shape}"),
print(f"Data types: {df.dtypes}"),


# %% [markdown]
# # Pandas Question 2
# 
# Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result.

# %%
passed_students = (df.passed == True) & (df.grade >= 80)
print(f"Students who passsed: {df[passed_students]}")

# %% [markdown]
# # Pandas Question 3
# 
# Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows).

# %%
grade_curved = df.grade + 5
df['grade_curved'] = grade_curved
print(df)

# %% [markdown]
# # Pandas Question 4
# 
# Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together.

# %%
name_upper = df['name'].str.upper()
df['name_upper'] = name_upper
df = df[['name', 'name_upper', 'grade', 'city', 'passed', 'grade_curved']]
print(df)

# %% [markdown]
# # Pandas Question 5
# 
# Group the DataFrame by "city" and compute the mean grade for each city. Print the result.

# %%
df = df.groupby('city').agg({'grade': 'mean'})

print(df)


# %% [markdown]
# # Pandas Question 6
# 
# Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.
# 

# %%
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)
df['city'] = df['city'].replace('Austin', 'Houston')
print(df[['name', 'city']])


# %% [markdown]
# # Pandas Question 7
# 
# Sort the DataFrame by "grade" in descending order and print the top 3 rows.

# %%
df = df.sort_values(by='grade', ascending=False)
print(df.head(3))

# %% [markdown]
# # NumPy Question 1
# 
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.

# %%
import numpy as np
import sys
arr1 = np.array([10, 20, 30, 40, 50])
print("Shape:", arr1.shape)
print("Data Type:", arr1.dtype)
print("Number of Dimensions:", arr1.ndim)



# %% [markdown]
# # NumPy Question 2
# 
# Create the following 2D array and print its shape and size (total number of elements).

# %%
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Shape:", arr.shape)
print("Size:", arr.size)

# %% [markdown]
# # NumPy Question 3
# 
# Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].

# %%
print(arr[0:2, 0:2])

# %% [markdown]
# # NumPy Question 4
# 
# Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.

# %%
print(np.zeros((3, 4)))
print(np.ones((2, 5)))



# %% [markdown]
# # NumPy Question 5
# 
# Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.
# 

# %%
arr2 = np.arange(0, 50, 5)

print(arr2)
print(arr2.shape)
print("Mean:", np.mean(arr2))
print("Sum:", np.sum(arr2))
print("Standard Deviation:", np.std(arr2))



# %% [markdown]
# # NumPy Question 6
# 
# Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.

# %%
np_array = np.random.normal(0, 1, 200)

print("Mean:", np_array.mean())
print("Standard Deviation:", np_array.std())

# %% [markdown]
# # Matplotlib Question 1
# 
# Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".

# %%
import matplotlib.pyplot as plt
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y)
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %% [markdown]
# # Matplotlib Question 2
# 
# Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.

# %%
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects, scores)
plt.title('Subject Scores')
plt.xlabel('subjects')
plt.ylabel('scores')
plt.show()

# %% [markdown]
# # Matplotlib Question 3
# 
# Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.

# %%
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color='black', label ='x1, y1')
plt.scatter(x2, y2, color='blue', label ='x2, y2')
plt.xlabel('x1, y1')
plt.ylabel('x2, y2')
plt.title('Scatter Plot')
plt.show()

# %% [markdown]
# # Matplotlib Question 4
# 
# Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing.

# %%
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
subjects = ['Math', 'Science', 'English', 'History']
scores = [88, 92, 75, 83]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(x, y)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('x vs y')

axes[1].bar(subjects, scores)
axes[1].set_xlabel('subjects')
axes[1].set_ylabel('scores')
axes[1].set_title('Subject Scores')

plt.tight_layout()
plt.show()

# %% [markdown]
# # Descriptive Stats Question 1
# 
# Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.

# %%
import numpy as np

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))

# %% [markdown]
# # Descriptive Stats Question 2
# 
# Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
# 
# 

# %%
import matplotlib.pyplot as plt
data_normal = np.random.normal(65, 10, 500)
plt.hist(data_normal, bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Scores')
plt.xlabel('Scores')
plt.ylabel('Distribution')
plt.show()

# %% [markdown]
# # Descriptive Stats Question 3
# 
# Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
# Hint: pass labels=["Group A", "Group B"] to plt.boxplot().
# 
# 

# %%
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], labels=['Group A', 'Group B'])
plt.title('Score Comparison')
plt.ylabel('Scores')
plt.show()

# %% [markdown]
# # Descriptive Stats Question 4
# 
# You are given two datasets: one normally distributed and one 'exponential' distribution.
# Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".
# 
# Then, add a comment in your code briefly noting which distribution is more skewed, and which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.

# %%
import numpy as np
import matplotlib.pyplot as plt

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.hist(skewed_data, bins=20, color='blue', edgecolor='black')

# %%


# %% [markdown]
# # Descriptive Stats Question 5
# 
# Print the mean, median, and mode of the following:
# 
# data1 = [10, 12, 12, 16, 18]
# data2 = [10, 12, 12, 16, 150]
# 
# Why are the median and mean so different for data2? Add your answer as a comment in the code.
# 
# 

# %%
data1 = [10, 12, 12, 16, 18] 
data2 = [10, 12, 12, 16, 150]

d1_median = print("data 1 median: ", np.median(data1))
d2_median = print("data 2 median: ", np.median(data2))

d1_mean = print("data1 mean: ", np.mean(data1))
d2_man = print("data2 mean: ", np.mean(data2))

# %% [markdown]
# The median is the same for dat1 and data2 because the median value is the same 12 for both datasets.  However, the mean is higher of data2 because it has an od value high of 150 which causes the mean value is higher.

# %%
%pip install scipy

# %% [markdown]
# # Hypothesis Question 1
# 
# Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.
# 
# 

# %%
from scipy import stats

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"t-statisctic:, {t_stat:.3f}")
print(f"p-value:, {p_val:.6f}")


# %% [markdown]
# # Hypothesis Question 2
# 
# Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05.

# %%
if p_value < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")

# %% [markdown]
# # Hypothesis Question 3
# 
# Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.

# %%
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
t_stat, p_val= stats.ttest_rel(before, after)
print(f"t-statistic:, {t_stat: .6f}")
print(f"p-value:, {p_val: .6f}")

# %% [markdown]
# # Hypothesis Question 4
# 
# Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.

# %%
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_val = stats.ttest_1samp(scores, 70)
print(f"t-statistic:, {t_stat:.3f}")
print(f"p-value:, {p_val:.6f}")

# %% [markdown]
# # Hypothesis Question 5
# 
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.

# %%
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
stats.ttest_ind(group_a, group_b, alternative="less")
print(f"p-value:, {p_val:.6f}")


# %% [markdown]
# # Hypothesis Question 6
# 
# Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.

# %% [markdown]
# Since the p-value is <=0.05 we reject the null hypotesis so there is strong evidence that the mean of the group a is less than the mean og the group b.

# %% [markdown]
# # Correlation Question 1
# 
# Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).

# %%
import pandas as pd
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
print(corr_matrix[0, 1])

# %% [markdown]
# # Correlation Question 2
# 
# Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.

# %%
from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r, p = pearsonr(x, y)
print("Correlation:", round(r, 2))
print("p-value:", round(p, 4))

# %% [markdown]
# # Correlation Question 3
# 
# Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.

# %%
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(df.corr())


# %% [markdown]
# # Correlation Question 4
# 
# Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.

# %%
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Negative Correlation")

# %% [markdown]
# # Correlation Question 5
# 
# Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".
# 
# Hint:
# 
# import seaborn as sns

# %%
%pip install seaborn

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# # Pipeline Question 1
# 
# A data pipeline is a sequence of processing steps where each step takes in data, transforms it, and passes the result to the next. You don't need a special framework to build one -- chaining plain functions together is often enough.
# 
# Given the array below, which contains some missing values scattered throughout:
# 
# arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
# Implement the following three functions and then connect them in a data_pipeline() function.
# 
# create_series(arr) : takes a NumPy array and returns a pandas Series with the name "values".
# clean_data(series) : takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
# summarize_data(series) -- takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode". For mode, use series.mode()[0] to get a single value.
# data_pipeline(arr) -- calls the three functions above in sequence and returns the summary dictionary.
# Call data_pipeline(arr) and print each key and its value from the result.
# 
# This is the last answer to put in warmups_01.py. Congrats!!!
# 
# The next question will be in prefect_warmup.py, but will implement the same functionality using Prefect instead of plain Python.
# 
# 

# %%
import numpy as np
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")    

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

def data_pipeline(arr):
    series = create_series(arr)
    clean_series = clean_data(series)
    summary = summarize_data(clean_series)
    return summary

results = data_pipeline(arr)
print("Data Summary:")
for key, value in results.items():
    print(f"{key.capitalize()}: {value:.2f}")






