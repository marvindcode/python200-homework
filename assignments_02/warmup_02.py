from turtle import color

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

predicted_salary = model.predict([[4]])
print("Predicted salary 4 years:", predicted_salary)
print("Slope:", model.coef_[0], "Intercept:", model.intercept_)

predicted_salary = model.predict([[8]])
print("Predicted salary 8 years:", predicted_salary)
print("Slope:", model.coef_[0], "Intercept:", model.intercept_)


# scikit-learn Question 2
# scikit-learn requires the feature array X to be 2D even when you only have one feature. Start with this 1D array:
# x = np.array([10, 20, 30, 40, 50])
# Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. Add a comment explaining, in your own words, why scikit-learn needs X to be 2D.

x = np.array([10, 20, 30, 40, 50])
print("Shape 1D array:", x.shape)

x_reshaped = x.reshape(-1, 1)
print("Shape 2D array:", x_reshaped.shape)

#Sickit-learn neeeds X to be 2D because it uses the data in tabular format, even if is only one variable.




# scikit-learn Question 3
# K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern as everything else in scikit-learn. Use the code below to generate a synthetic dataset with three natural clusters:
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt

# X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
# Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, and predict a cluster label for each point. Print the cluster centers (kmeans.cluster_centers_) and how many points fell into each cluster using np.bincount(labels).
# Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels. Save the figure to outputs/kmeans_clusters.png.

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print("Cluster centers:", kmeans.cluster_centers_)
print("Points fell into each cluster:", np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="black", marker='X', s=200, label="Cluster Centers")
plt.title("K-Means Clusters") 
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


# Linear Regression Question 1
# Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and cost on the y-axis. Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). Add a title "Medical Cost vs Age", label both axes, and save to outputs/cost_vs_age.png.
# Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the smoker variable?

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.7)
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.colorbar(label="Smoker")
plt.savefig("outputs/cost_vs_age.png")
plt.show()
#There are two distinct groups of points visible. The graph shows that smokers, showing in red, have higher medical costs than the people that do not smoke, showing in blue. The graph shows that the smoker variable has a significant impact on medical costs.


# Linear Regression Question 2
# Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42. Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.
X_train, X_test, y_train, y_test = train_test_split(age.reshape(-1, 1), cost, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Linear Regression Question 3
# Fit a LinearRegression model to your training data from Question 2. Print the slope and intercept. Then predict on the test set and print:
# •	RMSE: np.sqrt(np.mean((y_pred - y_test) ** 2))
# •	R² on the test set: model.score(X_test, y_test)
# Add a comment interpreting the slope in plain English -- what does it mean for medical costs?
model = LinearRegression()
model.fit(X_train, y_train)
print("Slope:", model.coef_[0], "Intercept:", model.intercept_)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

# The slope of $196.58 means that for each aditional year og age the medical cost increases by $196.58 if all else is constant. 


# Linear Regression Question 4
# Now add smoker as a second feature and fit a new model.
# X_full = np.column_stack([age, smoker])
# Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding the smoker flag help? Print both coefficients:
# print("age coefficient:    ", model_full.coef_[0])
# print("smoker coefficient: ", model_full.coef_[1])
# Add a comment interpreting the smoker coefficient: what does it represent in practical terms?

X_full = np.column_stack([age, smoker])
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, cost, test_size=0.2, random_state=42)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
y_pred_full = model_full.predict(X_test_full)
r2_full = r2_score(y_test_full, y_pred_full)    
print("R² with age only:", r2)
print("R² with age and smoker:", r2_full)
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

#The smoker coefficient of $14,538.04 means that being a smoker increases the medical cost by $14,538.04 compared to non-smokers, holding age constant. That means that being a smoker impacts more on medical costs than age.  That relationship shows a stronger R2 value of .77 compared to the R2 value of 0.07 when only age was used in the model.



# Linear Regression Question 5
# A predicted vs actual plot is a standard tool for evaluating regression models. Each test observation becomes a dot: the model's prediction goes on the x-axis, the true value goes on the y-axis. A perfect model would place every point on the diagonal line where predicted equals actual.
# Using the two-feature model from Linear Regression Question 4, create this plot for the test set. Add a diagonal reference line, a title "Predicted vs Actual", labeled axes, and save to outputs/predicted_vs_actual.png.
# Add a comment: what does it mean when a point falls above the diagonal? What about below?

plt.figure(figsize=(8, 8))
plt.scatter(y_pred_full, y_test_full, alpha=0.7, label="Two-feature model")
min_val = min(y_test_full.min(), y_pred_full.min())
max_val = max(y_test_full.max(), y_pred_full.max())

plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle='--', label="Perfect predictions")

plt.title("Predicted vs Actual")    
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")

plt.legend()
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()  

# If a point falls above the diagonal means that the actual value is higher than the predicted, that is called an understimation. If a point falls below the diagonal means that the predicted value is higher than the actual, that is called an overestimation.  But if we average those differences the errors cancel each other out.
