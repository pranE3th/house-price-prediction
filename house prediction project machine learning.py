import kagglehub
import pandas as pd  # For 'DATA CLEANING AND PRE-PROCESSING' step.
import numpy as np   # same as above.
from scipy.stats import f_oneway
import matplotlib.pyplot as plt # For 'EDA' step.
import seaborn as sns           # For 'EDA' step.
import shap                     # For 'Predictions' and 'Visualisations'(step-7)
from google.colab import files
from IPython.core.display import HTML
from scipy.stats import ttest_ind # For 'Feature Selection' step(for
# statistical tests).
from scipy.stats import kruskal # For 'Feature Selection' step(for
# statistical tests for 'MULTICATEGORIAL FEATURE' i.e., 'Furnishing status').

from sklearn.model_selection import train_test_split # For 'Wrapper Method'.
from sklearn.feature_selection import RFE            # same
from sklearn.ensemble import RandomForestRegressor   # same
from sklearn.feature_selection import SequentialFeatureSelector  # same

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import metrics
from sklearn.inspection import PartialDependenceDisplay # For 'Predictions' and
# 'Visualisations'(step-7)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV # For step - 5(Hyperparameter
# Tuning)

import os

!pip install flask flask-ngrok apscheduler pandas scikit-learn joblib



'''
Factors influence house pricing(hp) are:
    1. If area ^ then hp ^
    2. If total_rooms value ^ then hp ^
    3. If mainroad == yes, then hp ^
    4. If aircond. and/or waterheater == yes, then hp ^
    5. If parking or/and basement  == yes, then hp ^
    6. If prefarea == yes, then hp ^
    7. hp of (furnished > semi-furnished > unfurnished)
 '''




# STEP - 1(DATA COLLECTION(from kaggle))

# Download the latest version of the dataset
path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

# List the contents of the directory
csv_files = []
for f in os.listdir(path):
  print('this---->',f)
  if f.endswith('.csv'):
    csv_files.append(f)

# Print the names of the CSV files found
print("CSV files found:", csv_files)


# Load the first CSV file into a DataFrame (adjust if needed)
if csv_files:
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    # Load the first CSV file
    # Print a specific column (replace 'column_name' with the actual column name)
    # print(df['price'])
    #print(df)
else:
    print("No CSV files found in the directory.")



# STEP - 2(DATA PRE-PROCESSING)

# Renaming the column names/attributes with better naming convention.
df = df.rename(columns={'hotwaterheating':'waterheater',
                        'waterheater':'water_heater',
                        'airconditioning':'air_conditioning',
                        'prefarea':'preffered_area',
                        'furnishingstatus':'furnishing_status'
                        })



# Combining bedrooms, bathrooms and guestroom as total_rooms.

if ('bedrooms' in df) and ('bathrooms' in df) and ('guestroom' in df):
  for i in range(len(df['guestroom'])):
    Total_rooms = df['bedrooms'][i] + df['bathrooms'][i]
    if df['guestroom'][i] == 'yes':
      Total_rooms += 1


print(Total_rooms)







df = df.drop(['bedrooms','bathrooms','guestroom'], axis = 1)

# Adding newly created total_rooms column to the existing dataframe.
df['Total_rooms'] = Total_rooms


TR = df.pop('Total_rooms')

df.insert(2,'Total_rooms',TR)

price = df.pop('price')

df.insert(10,'price',price)

stories = df.pop('stories')



df.columns = df.columns.str.capitalize()

df




# df['hotwaterheating'][0]



# STEP - 3(Exploritary Data Analysis(EDA) or Visualizations)

# Knowing about the data structure and summary about the dataset.

df.head()

df.info()

df.describe()

sns.histplot(df['Price'], kde=True)


# Checking if price col is skewed or not

sns.histplot(df['Price'], kde=True)
plt.title("Distribution of House Prices")
plt.show()


#df['log_price'] = np.log1p(df['Price'])
#
#df['log_Area'] = np.log1p(df['Area'])
#
#sns.histplot(df['log_price'], kde=True)
#plt.title("Log-Transformed Price Distribution")
#plt.show()

# Relationships between price and other numerical value features.

# 1. price vs area(between their log's for having small values)

'''
sns.scatterplot(x=df['log_price'], y=df['log_Area'])
plt.title('Price vs Area')
plt.xlabel('Price')
plt.ylabel('Area')
'''

sns.scatterplot(x=df['Area'], y=df['Price'])
sns.regplot(x=df['Area'], y=df['Price'], scatter=False, line_kws={"color": "red"})
plt.title('Price vs Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# 2. price vs total_rooms
'''
sns.bubblechart(x=df['Total_rooms'], y=df['Price'])
plt.title('Price vs Total_rooms')
plt.xlabel('Price')
plt.ylabel('Total_rooms')
plt.show()'''

''' 3. price vs mainroad(stripplot will showcases the relation better incase of
categorial variable(in this case mainroad)) '''

sns.stripplot(x=df['Mainroad'], y=df['Price'])
plt.title('Price vs Mainroad')
plt.xlabel('Price')
plt.ylabel('Mainroad')
plt.show()

# 4. price vs waterheater

sns.boxplot(x=df['Waterheater'], y=df['Price'])
plt.title('Price vs Waterheater')
plt.xlabel('Price')
plt.ylabel('Waterheater')
plt.show()

# 5. price vs air_conditioning

sns.violinplot(x=df['Air_conditioning'], y=df['Price'])
plt.title('Price vs Air_conditioning')
plt.xlabel('Price')
plt.ylabel('Air_conditioning')
plt.show()

# 6. price vs pref_area

sns.barplot(x=df['Preffered_area'], y=df['Price'])
plt.title('Price vs Preffered_area')
plt.xlabel('Price')
plt.ylabel('Preffered_area')
plt.show()

# 7. price vs furnishing status

sns.barplot(x=df['Furnishing_status'], y=df['Price'])
plt.title('Price vs Furnishing_status')
plt.xlabel('Price')
plt.ylabel('Furnishing_status')
plt.show()

df


'''
plt.figure(figsize=(10, 6))
plt.scatter(
    x=df['Area'],
    y=df['Price'],
    s=df['Total_rooms'] * 10,  # Scale bubble size based on Total_rooms
    alpha=0.6,
    c=df['Parking'],  # Optional: color bubbles based on Parking spaces
    cmap='viridis'
)
plt.colorbar(label='Parking Spaces')
plt.title('Bubble Chart: Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

'''

# *** Revision

# STEP - 4(Feature Selection -> Model Selection and Training -> Model Evaluation)

# Method - 1(Filter Method(correlation and statistical tests))

# a. Correlation

''' Both Numerical Dataset and Categorical Dataset are unneccessary. '''

# Numerical Dataset
Numerical_df = df.select_dtypes(include=['number'])

# Categorical Dataset

Categorical_df = df.select_dtypes(include=['object', 'category'])

print("Numerical Features:", Numerical_df.columns.tolist())
print("Numerical Features:", Categorical_df.columns.tolist())

corr_matrix = Numerical_df.corr()

# Visualize the correlations
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Select features with high correlation to Price
high_corr_features = corr_matrix['Price'][corr_matrix['Price'].abs() > 0.5].index
print("Highly correlated features with Price:", high_corr_features)




Numerical_df

Categorical_df




# *** Revision

# b. Statistical Tests



# 1. Price vs Mainroad.

# Split `Price` based on `Mainroad`

mainroad_yes = df.loc[df['Mainroad'] == 'yes']['Price']
mainroad_no = df.loc[df['Mainroad'] == 'no']['Price']

# Perform t-test
t_stat, p_value = ttest_ind(mainroad_yes, mainroad_no)

# Output results
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference: Reject the null hypothesis.")
else:
    print("No significant difference: Fail to reject the null hypothesis.")

'''
CONCLUSION:
t-statistic: 7.245125201307269,
p-value: 1.4901041488906289e-12.
So there is impact of 'Mainroad' factor on target var
(Significant difference: Reject the null hypothesis).

'''

# 2. Price vs Basement.

basement_yes = df.loc[df['Basement'] == 'yes']['Price']
basement_no = df.loc[df['Basement'] == 'no']['Price']

# Perform t-test
t_stat, p_value = ttest_ind(basement_yes, basement_no)

# Output results
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference: Reject the null hypothesis.")
else:
    print("No significant difference: Fail to reject the null hypothesis.")

'''
CONCLUSION:
t-statistic: 4.437180316396756,
p-value: 1.1041051901314538e-05.
So there is impact of 'Basement' factor on target var
(Significant difference: Reject the null hypothesis).

'''

# 3. Price vs Waterheater.

waterheater_yes = df.loc[df['Waterheater'] == 'yes']['Price']
waterheater_no = df.loc[df['Waterheater'] == 'no']['Price']

# Perform t-test
t_stat, p_value = ttest_ind(waterheater_yes, waterheater_no)

# Output results
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference: Reject the null hypothesis.")
else:
    print("No significant difference: Fail to reject the null hypothesis.")


'''
CONCLUSION:
t-statistic: 2.178272173683366,
p-value: 0.029815238966018866.
So there is impact of 'Waterheater' factor on target var
(Significant difference: Reject the null hypothesis).

'''

# 4. Price vs Air_conditioner.

Air_conditioning_yes = df.loc[df['Air_conditioning'] == 'yes']['Price']
Air_conditioning_no = df.loc[df['Air_conditioning'] == 'no']['Price']

# Perform t-test
t_stat, p_value = ttest_ind(Air_conditioning_yes, Air_conditioning_no)

# Output results
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference: Reject the null hypothesis.")
else:
    print("No significant difference: Fail to reject the null hypothesis.")


'''
CONCLUSION:
t-statistic: 11.839033782035843,
p-value: 6.310969853530074e-29.
So there is impact of 'Waterheater' factor on target var
(Significant difference: Reject the null hypothesis).

'''

# 5. Price vs Preffered_area.

Preffered_area_yes = df.loc[df['Preffered_area'] == 'yes']['Price']
Preffered_area_no = df.loc[df['Preffered_area'] == 'no']['Price']

# Perform t-test
t_stat, p_value = ttest_ind(Preffered_area_yes, Preffered_area_no)

# Output results
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference: Reject the null hypothesis.")
else:
    print("No significant difference: Fail to reject the null hypothesis.")


'''
CONCLUSION:
t-statistic: 8.139941413971995,
p-value: 2.718374467072544e-15.
So there is impact of 'Waterheater' factor on target var
(Significant difference: Reject the null hypothesis).

'''

# 6. Price vs Furnishing_status.

# Split `Price` by `Furnishing_status`
furnished = df.loc[df['Furnishing_status'] == 'furnished']['Price']
semi_furnished = df.loc[df['Furnishing_status'] == 'semi-furnished']['Price']
unfurnished = df.loc[df['Furnishing_status'] == 'unfurnished']['Price']

# Perform Kruskal-Wallis test(because of more than 2 categories and instead
# ANOVA used this because is price is not normal).

h_stat, p_value = kruskal(furnished, semi_furnished, unfurnished)

# Output results
print("H-statistic:", h_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference: Reject the null hypothesis.")
else:
    print("No significant difference: Fail to reject the null hypothesis.")

'''
CONCLUSION:
H-statistic: 69.58292445963657,
p-value: 7.767118592501638e-16.
So there is impact of 'Furnishing_status' factor on target var
(Significant difference: Reject the null hypothesis).

'''

# *** Revision

# Method - 2(Wrapper Method())

for i in range(len(df['Price'])):
   if df['Mainroad'][i] == 'yes':
      df['Mainroad'][i] = 1
   else:
      df['Mainroad'][i] = 0

   if df['Basement'][i] == 'yes':
      df['Basement'][i] = 1
   else:
      df['Basement'][i] = 0

   if df['Waterheater'][i] == 'yes':
      df['Waterheater'][i] = 1
   else:
      df['Waterheater'][i] = 0

   if df['Air_conditioning'][i] == 'yes':
      df['Air_conditioning'][i] = 1
   else:
      df['Air_conditioning'][i] = 0

   if df['Preffered_area'][i] == 'yes':
      df['Preffered_area'][i] = 1
   else:
      df['Preffered_area'][i] = 0

   if df['Furnishing_status'][i] == 'furnished':
      df['Furnishing_status'][i] = 1
   elif df['Furnishing_status'][i] == 'semi-furnished':
      df['Furnishing_status'][i] = 0.5
   else:
      df['Furnishing_status'][i] = 0

df

X = df.drop(columns=['Price'])
y = df['Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Define the model (Random Forest Regressor in this case)
model = RandomForestRegressor(random_state=42)

# 1. Perform Recursive Feature Elimination (RFE):

rfe = RFE(estimator=model, n_features_to_select=3)  # Select top 3 features
rfe.fit(X_train, y_train)

# Results
rfe_selected_features = X_train.columns[rfe.support_]
print("RFE Selected Features:", rfe_selected_features)

# Train and evaluate the model using selected features
X_train_rfe = X_train[rfe_selected_features]
X_test_rfe = X_test[rfe_selected_features]

model.fit(X_train_rfe, y_train)
y_pred_rfe = model.predict(X_test_rfe)

# Calculate R² and RMSE for RFE:
r2_rfe = r2_score(y_test, y_pred_rfe)
rmse_rfe = np.sqrt(mean_squared_error(y_test, y_pred_rfe))
print(f"R² (RFE): {r2_rfe:.4f}, RMSE (RFE): {rmse_rfe:.4f}")

'''
# Ranking of all features
feature_ranking = pd.DataFrame({
    'Feature': X_train.columns,
    'Rank': rfe.ranking_
}).sort_values(by='Rank')

print("Feature Ranking:\n", feature_ranking)
'''

# 2. Perform Sequential Feature Selection (Forward Selection):

# Apply Forward Sequential Feature Selection
sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=3,
                                direction='forward', scoring='r2', cv=5)
sfs.fit(X_train, y_train)

# Get selected features
sfs_selected_features = X_train.columns[sfs.get_support()]
print("SFS Selected Features:", sfs_selected_features)

# Train and evaluate the model using selected features
X_train_sfs = X_train[sfs_selected_features]
X_test_sfs = X_test[sfs_selected_features]

model.fit(X_train_sfs, y_train)
y_pred_sfs = model.predict(X_test_sfs)

# Calculate R² and RMSE for SFS:

r2_sfs = r2_score(y_test, y_pred_sfs)
rmse_sfs = np.sqrt(mean_squared_error(y_test, y_pred_sfs))
print(f"R² (SFS): {r2_sfs:.4f}, RMSE (SFS): {rmse_sfs:.4f}")

# Comparison of Results
print("\nComparison of R² and RMSE:")
print(f"R² (RFE): {r2_rfe:.4f}, RMSE (RFE): {rmse_rfe:.4f}")
print(f"R² (SFS): {r2_sfs:.4f}, RMSE (SFS): {rmse_sfs:.4f}")

if r2_rfe > r2_sfs and rmse_rfe < rmse_sfs:
    print("RFE provides better performance overall.")
elif r2_sfs > r2_rfe and rmse_sfs < rmse_rfe:
    print("SFS provides better performance overall.")
else:
    print('''The methods perform similarly. Consider other factors like
    computation time or feature interpretability.''')


selected_features = rfe_selected_features
print("Final Selected Features by RFE:", selected_features)


# Train model with RFE-selected features
X_train_final = X_train[selected_features]
X_test_final = X_test[selected_features]

model.fit(X_train_final, y_train)
y_pred_final = model.predict(X_test_final)

# Evaluate performance
final_r2 = r2_score(y_test, y_pred_final)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

print(f"Final Model Performance (RFE Features): R² = {final_r2:.4f}, RMSE = {final_rmse:.4f}")

# *** Revision

# STEP - 5(Hyperparameter Tuning/Optimising):


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(X_train_final, y_train)

print("Best Parameters:", grid_search.best_params_)


# *** Revision

# STEP - 6(Visualizations/Interpreting Results(Predictions done in step - 4)):

# Get feature importances from the model trained with RFE features
importances = model.feature_importances_
feature_names = selected_features

print(feature_names)

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance for House Pricing Prediction')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()


#Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_final)

# Summary plot
shap.summary_plot(shap_values, X_test_final, feature_names=selected_features)


# Generate Partial Dependence Plots (PDPs) for selected features
features_to_plot = [0, 1, 2]  # Indices of features to plot (e.g., 0 = 'Area', 1 = 'Total_rooms')
PartialDependenceDisplay.from_estimator(model, X_test_final,
                                        features=features_to_plot, feature_names=selected_features)

# Display the plot
plt.show()

'''
shap.initjs()

# Generate SHAP force plot for the first prediction in the test set
shap.force_plot(explainer.expected_value, shap_values[0], X_test_final.iloc[0, :], feature_names=selected_features)
'''

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values for the test data
shap_values = explainer.shap_values(X_test)

shap.initjs()

# Save force plot as HTML
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[0],  # SHAP values for the first prediction
    X_test.iloc[0, :],  # Feature values for the first test instance
    feature_names=X.columns  # Optional: Include feature names
)

# Save the force plot as an HTML file
shap.save_html('force_plot.html', force_plot)

'''
(Optional) View HTML File Without Downloading If you prefer to view the plot
without downloading the file, you can directly render it in a Jupyter notebook
or Colab environment using an HTML iframe. Add the following code after saving
the file:
below is the approach:
'''

# Render the HTML file in Colab
HTML(filename='force_plot.html')


print(model)

# Replace 'model' with the name of your trained model
joblib.dump(model, 'house_price_model.pkl')

print("Model saved as house_price_model.pkl")


from google.colab import files
files.download('house_price_model.pkl')




'''
Steps from 3 to 8 are as follows:

-> After EDA(choose better visualizations, independent var = price),
-> observe and know which attributes influences price more significantly(read liner
desc clearly),
-> divide the dataframe into two df's i.e, training and testing(see at what %),
-> know the process of regression and how to implement it for our project,
-> know the performance metrics and use it for the our model to know its perf.
level(model evaluation),
-> predictions i.e, testing the model with test dataframe we had created,
at last, the final step is deployement.

'''



# print(df.keys)
# print(df.values)