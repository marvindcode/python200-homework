#The fields of the dataset student_performance_math.csv are delimeted by semicolons. The variables G1,G2 and G3 are separeted by semicolons.

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from scipy import stats
from scipy.stats import pearsonr

#Task 1: Load and Explore
df = pd.read_csv("student_performance_math.csv", sep=";")
print(df.head(5))
print(df.shape)
print(df.dtypes)

plt.hist(df["G3"], bins=20, edgecolor="black")
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Frequency")
plt.savefig("outputs/g3_distribution.png")
plt.show()

#Task 2: Preprocess the Data
df1 = df.drop(columns=["G1", "G2"], inplace=True)
df1 = df[df["G3"] != 0]

print(df1.shape)

plt.hist(df1["G3"], bins=20, edgecolor="black")
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Frequency")
plt.savefig("outputs/g3_distribution_normal.png")
plt.show()


#Reasoning: Taking out the students that didn't take the exam allow us to evaluate the students that took the exam and how the other variables affect the grade.  Running the histogram without the students that didn't took the exam shows a more normal distribution of the grade for hte G3 final period grade.

# Then convert the yes/no columns to 1/0 and the sex column to 0/1.

# Now check something interesting before moving on. Compute the Pearson correlation between absences and G3 on both the original dataset and the filtered one, and print both values. The difference is striking. Add a comment explaining why filtering changes the result: what were students with G3=0 doing in the original data that made absences look like a weak predictor? You might want to explore scatter plots to help understand this.

categorical_cols = df1.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    le = LabelEncoder()
    df1[col] = le.fit_transform(df1[col])
    # df1[col] = df1[col].map({"yes": 1, "no": 0, "M": 1, "F": 0})

print(df1.head(5))
print(df1.info())

correlation_original = pearsonr(df["absences"], df["G3"])
correlation_filtered = pearsonr(df1["absences"], df1["G3"])
print("Correlation (original):", correlation_original)
print("Correlation (filtered):", correlation_filtered)

plt.scatter(df["absences"], df["G3"], alpha=0.7)
plt.title("Absences and Final Grade (Original)")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/absences_g3_original.png")
plt.show()

plt.scatter(df1["absences"], df1["G3"], alpha=0.7)
plt.title("Absences and Final Grade (Filtered)")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/absences_g3_filtered.png")
plt.show()

#The correlation between absences and G3 before removiing the students with absences is 0.034, which is a low correlation. That means that the grades are not impacted by the number of absences. But after removing the students with G3=0, the correlation is -0.2131, which is a stronger negative correlation. That means that the grades are impacted by the number of absences, as the more absences a student has, the lower their final grade tends to be.


# Task 3: Exploratory Data Analysis

# Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset, and print them sorted from most negative to most positive. Which feature has the strongest relationship with G3? Are any results surprising?

# Then create at least two visualizations of your own choosing and save them to outputs/. Use your judgment from previous weeks of data engineering to guide your use of plots. Use the correlation results to guide you -- what relationships seem worth a closer look? Add a comment for each plot describing what you see.

for col in df1.select_dtypes(include=[np.number]).columns:
    if col != "G3":
        correlation, _ = pearsonr(df1[col], df1["G3"])
        print(f"Correlation between {col} and G3: {correlation}")

print(df1.corr()["G3"].sort_values())

correlation_matrix = df1.corr()

plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation Coefficient")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Matrix")
plt.savefig("outputs/correlation_matrix.png")
plt.show()

#With this graph we can see the correlation between all the variables.  That allows us to see that the variables with the strongest correlation with G3. This graph is useful because shows which are more  correlated eventhoug the correlation is negative. For example, the variable with the strongest correlation with G3 is Medu, with a correlation of 0.19, this explains the relationship betweeen the Mother' education level and G3.  Mothers with high level could motivate the student.  On the inverse relationship, the variable with strongest correlation is failures, how this is explain is that with more failures the final period grade G# is lower.


plt.figure(figsize=(8, 6))
plt.scatter(df1["failures"], df1["G3"], alpha=0.7)
plt.title("Failures and Final Grade")
plt.xlabel("Number of Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/failures_g3.png")
plt.show()

#On this graph we can see the relationship between the number of failures and the final grade. The relationshiop is inverse which it means that students with more failures tend to have lower final grades. 

# Task 4: Baseline Model

# Build the simplest possible model: use failures alone to predict G3. Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print the slope, RMSE, and R² on the test set.
# Add a comment: given that grades are on a 0-20 scale, what do the slopes and RMSE tell you in plain English? Is R² better or worse than you expected from exploratory data analysis?

y = df1["G3"]
X = df1[["failures"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Slope:", model.coef_[0], "Intercept:", model.intercept_)
y_pred = model.predict(X_test) 
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R²:", r2)    


#The slope of -1.4275 is telling us that for each additional failure, the final grade decrease by 1.4275, leaving all constant.  The RMSE of 2.96 shows that the prediction of the model is off by an average of 2.96 points. The R² value of 0.08949 indicates that only about 8.9% of the variance in the final grades can be explained by the number of failures alone.  We can conclude that this model is not a good predictor of the final grade.



# Task 5: Build the Full Model

# Now build a regression model using all of the numeric and binary features from the Feature Guide:


# Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print both train R² and test R², as well as RMSE on the test set. Compare the test R² to your baseline from Task 4 -- how much does adding more features help?
# Print each feature name alongside its coefficient:

# for name, coef in zip(feature_cols, model.coef_):
#     print(f"{name:12s}: {coef:+.3f}")
# Look carefully at the coefficients. Sort them mentally from largest to smallest. Are any signs (positive or negative) surprising given what you know about the data? For any surprising result, add a comment with your best explanation. Then compare train R² to test R² -- are they close, or is there a gap? What does that tell you about the model?

# Finally, add a comment answering: if you were deploying this model in production, which features would you keep and which would you drop? Justify your choices based on what you see in the numbers.

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df1[feature_cols].values
y = df1["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)
print("Train R²:", r2_train)
print("Test R²:", r2_test)
print("RMSE:", rmse)

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

#The R2 from the baseline model was 0.089, but the R2 from the full model is 0.25, which means that more variables explains the variance in the final grades. The model improved with more features. This means that the full model explains 25% of the variance in the final grades, compared to only 8.9% with the baseline model.

#To evaluate the coefficientes, we can evaluate the most negative and the most positive. The most negative is schoolsup, which means that studentes with more support from the scholl have lower final grades, meaning that the students that have more support is because they eed it to get better grades. The most positive is internet, which means that students that have access to internet can reaserh more or have more resources to practice.

#The coefficient that surprised me the most is study time, which is positive but is not the highest, I would think that students that study more get better grades, that means that is the student probably study more is because is more difficult the subject for them.

# The train R² is 0.17 and the test R² is 0.15, are pretty clos which means there is no overfitting.    

# If I were deploying this model in production, I would keep the two features with the highest positive and negative coefficients, such as internet and higher and schoolsup and failures. I think using the features with higher numbers have more weight on the prediction variable.  The ones that I would drop would be the clossest to zero, like freetime and activities because they have less impact on the final grade.  


# Task 6: Evaluate and Summarize

# A useful way to evaluate a regression model visually is a predicted vs actual plot. This is a scatter plot where each point in the test set becomes a dot, with the model's prediction (y_hat) on the x-axis and the true value (y) on the y-axis. If the model were perfect, every point would fall exactly on the diagonal (predicted = actual). Clusters or curves away from the diagonal reveal systematic errors that RMSE alone won't show you. Random scattering around the diagonal is expected, and acceptable, prediction error.
# Create this plot for your test set. Add a diagonal reference line (for y=y_predicted), a title "Predicted vs Actual (Full Model)", labeled axes, and save to outputs/predicted_vs_actual.png. Add a comment: does the model seem to struggle more at the high end, the low end, or is error roughly uniform across grade levels? What does a value above or below the diagonal mean?
# Then write a plain-language summary in your comments statements covering:

# The size of the filtered dataset and the test set
# The RMSE and R² of your best model in plain language -- on a 0-20 scale, what does a typical prediction error actually mean?
# Which two features have the largest positive and largest negative coefficients, and what those mean
# One result that surprised you
# Neglected Feature: The Power of G1

# Add G1 (first period grade) as a feature to the full model from Task 5 and refit. We kept it out because it is so powerful. Print the new test R². The jump will be large -- from roughly 0.30 to somewhere around 0.80.
# Add a comment addressing these questions: does a high R² here mean G1 is causing G3? Is this a useful model for identifying students who might struggle? What might educators need to do if they wanted to intervene early, before G1 is even available?


df1 = pd.read_csv("student_performance_math.csv", sep=";")

df_clean = df1[df1["G3"] > 0].copy()

df_clean["schoolsup"] = df_clean["schoolsup"].map({"yes": 1, "no": 0})
df_clean["internet"] = df_clean["internet"].map({"yes": 1, "no": 0})
df_clean["higher"] = df_clean["higher"].map({"yes": 1, "no": 0})
df_clean["activities"] = df_clean["activities"].map({"yes": 1, "no": 0})
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
    "internet", "sex", "freetime", "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = model.score(X_test, y_test)

print("Test R² (full model):", r2_test)
print("RMSE (full model):", rmse)

plt.figure(figsize=(8, 8))
plt.scatter(y_pred, y_test, alpha=0.7)

min_val = min(y_pred.min(), y_test.min())
max_val = max(y_pred.max(), y_test.max())

plt.plot([min_val, max_val], [min_val, max_val],color="red", linestyle="--", label="Perfect Prediction")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Final Grade (G3)")
plt.ylabel("Actual Final Grade (G3)")
plt.legend()
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()


feature_cols_g1 = feature_cols + ["G1"]

X_g1 = df_clean[feature_cols_g1].values
y_g1 = df_clean["G3"].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(X_g1, y_g1, test_size=0.2, random_state=42)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

r2_test_g1 = model_g1.score(X_test_g1, y_test_g1)

print("Test R² with G1:", r2_test_g1)

#The results adding G1 increase the R2 from 0.15 to 0.749.  That means that the first period grade is strong predictor of the final grade.  The G1 makes the model stronger but that doesn't mean is a perfect predictor oof the final grade of Grade 3, there is other variables that we evalkuated that could have more meaning for the model.
