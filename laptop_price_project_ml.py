
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('laptop.csv')

"""# Data Exploration and Understanding"""

# EDA
data.sample(10)

data.isnull().sum()

data.shape

# Handling NaN/duplicates

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.drop(columns=['Unnamed: 0.1','Unnamed: 0'], axis=1, inplace=True)

sns.histplot(x='Price', data=data, kde=True, color='red')
plt.title('Laptop Price Distribution')
plt.show()

# countplots for different features like Brand, Type, OS etc

def drawplot(col):
    plt.figure(figsize=(8,5))
    sns.countplot(data[col], palette='plasma')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.show()

toview = ['Company', 'TypeName','OpSys']
for col in toview:
    drawplot(col)

# Price variation on each Laptop Brand

sns.color_palette(palette='Accent')
plt.figure(figsize=(8,5))
plt.title('Campany/Brand Vs Laptop Price')
sns.barplot(x = data['Company'],y = data['Price'], palette='mako') ##ci=None
plt.xticks(rotation = 'vertical')
plt.show()

# Impact of OS on Laptop Price

sns.color_palette(palette='Accent')
plt.figure(figsize=(8,5))
plt.title('Operating System Vs Laptop Price')
sns.boxplot(x='OpSys', y='Price', data=data, palette='Accent')
plt.xticks(rotation = 'vertical')
plt.show()

# No OS, Chrome OS, Linux/Android laptops are comparitively in a cheaper price range
# Apple/MacOS & Windows 7 laptops have a price range: 50K-150K
# Outliers in Windows-10 Laptops due to higher specifications like RAM, GPU, Memory, ScreenResolution, Inches etc

# converting datatype of Ram & Weight Columns from Object to int

data['Ram']= data['Ram'].str.replace('GB','')
data['Ram']= data['Ram'].astype('int32')

# Impact of RAM on Laptop Price

plt.figure(figsize=(8,5))
sns.scatterplot(x='Ram', y='Price', data=data, c='r')
plt.title('RAM Vs Laptop Price')
plt.show()

# checks if any of the values in the Weight column contain non-numeric characters

data['Weight'].str.contains('[^0-9.]').any()

# replacing strings from Weight column

data['Weight'] = data['Weight'].str.replace('kg','')
data['Weight'] = data['Weight'].str.replace('[^0-9.]', '')

# Remove rows with non-numeric values:

data = data[data['Weight'].astype(str) != '']
data['Weight'].fillna(0, inplace=True)

# converting dtype to Float

data['Weight'] = data['Weight'].astype('float32')

# Impact of Weight on Laptop Price

plt.figure(figsize=(8,5))
sns.scatterplot(x='Weight', y='Price', data=data, c='g')
plt.title('Weight Vs Laptop Price')
plt.show()

# Identifying incorrect 'Inches' data

data[data['Inches'].str.contains('[^0-9.]')]

# Imputation to Inches column

data['Inches'] = data['Inches'].str.replace('?','15.6')

#data.loc[(data.Inches == '?'), ['Inches']] = 15.6
data['Inches'] = data['Inches'].astype('float32')

#data[data['ScreenResolution']=='Full HD 1920x1080'].mean()

data['Company'].value_counts()

data.info()

data.corr()['Price']

data['ScreenResolution'].value_counts()

# Feature Engineering
# Creating a new feature: 'Touchscreen' from ScreenResoltion column

data['TouchScreen'] = data['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

# Creating a new feature: 'IPS' from ScreenResoltion column

data['IPS'] = data['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

data.sample(10)

# Impact of TouchScreen & IPS display on Laptop Price

plt.figure(figsize=(6,4))
sns.barplot(x='TouchScreen', y='Price', data=data, hue='IPS')
plt.show()

# IPS display in No Touchscreen category have higher Price. ie,IPS display has more impact on Price

# Splitting screen resolution to x & y -resolutions

res_data = data['ScreenResolution'].str.split('x',n = 1,expand=True)

# exatrcting digits only from 1st column of the new dataframe
data['x_res'] = res_data[0].str.replace(',','').str.findall(r'(\d+.?\d+)').apply(lambda x:x[0])

data['y_res'] = res_data[1]

# converting dtype to INT
data['x_res'] = data['x_res'].astype('int')
data['y_res'] = data['y_res'].astype('int')

# Creating new feature 'PPI'(pixels per inch), PPI = root(x_res^2 + y_res^2) / Inches
data['PPI'] = (((data['x_res']**2 + data['y_res']**2))**0.5/data['Inches']).astype('float')

# dropping unwanted columns
data.drop(columns=['ScreenResolution','Inches','x_res','y_res'],inplace=True)

data.head()

# Impact of display-PPI on Laptop Price

plt.figure(figsize=(8,5))
sns.regplot(x='PPI', y='Price', data=data, ci=None)
plt.title('PPI Vs Laptop Price')
plt.show()

# Extracting CPU names

data['CPU_name'] = data['Cpu'].apply(lambda text:" ".join(text.split()[:3]))

# processing CPU names

def processortype(text):
  if text=='Intel Core i3' or text=='Intel Core i5' or text=='Intel Core i7':
    return text
  else:
    if text.split()[0]=='Intel':
      return 'Others Intel'
    else:
      return 'AMD'


data['CPU_name'] = data['CPU_name'].apply(lambda text:processortype(text))
data.head()

data['CPU_name'].value_counts()

data['Memory'].value_counts()

# Imputation to Memory column

data['Memory'] = data['Memory'].str.replace('?','256GB SSD')

# remove decimal space i.e, 1.0 TB to 1TB
# replace the word GB with "",TB with "000", split the word across the "+" character

data['Memory'] = data['Memory'].astype(str).replace('\.0','',regex = True)
data['Memory'] = data['Memory'].str.replace('GB','')
data['Memory'] = data['Memory'].str.replace('TB','000')
newdata = data['Memory'].str.split("+",n = 1,expand = True)
newdata

data['MemType'] = newdata[0]
data['MemType'] = data['MemType'].str.strip()

# Creating separate column for each memory type

def create_mem(value):
    data['Mem_'+value] = data['MemType'].apply(lambda x:1 if value in x else 0)


mem_list = ['HDD','SSD','Hybrid','Flash Storage']

for value in mem_list:
    create_mem(value)

data.head()

# removing characters

data['MemType'] = data['MemType'].str.replace(r'\D','')
data['MemType'].value_counts()

# for MemType column -2

data['MemType_2'] = newdata[1]
data['MemType_2'] = data['MemType_2'].str.strip()

# Creating separate column for each memory type-2 column

def create_mem(value):
    data['Mem_2'+value] = data['MemType_2'].apply(lambda x:1 if value in x else 0)


mem_list = ['HDD','SSD','Hybrid','Flash Storage']
data['MemType_2'] = data['MemType_2'].fillna("0")

for value in mem_list:
    create_mem(value)

# removing characters

data['MemType_2'] = data['MemType_2'].str.replace(r'\D','')
data['MemType_2'].value_counts()

# Converting 2 Memory type columns to Int

data['MemType']=data['MemType'].astype('int')
data['MemType_2']=data['MemType_2'].astype('int')

# Multiplying new MemType(int) columns

data['HDD'] = ( data['MemType'] * data['Mem_HDD'] + data['MemType_2'] * data['Mem_2HDD'] )

data['SSD'] = ( data['MemType'] * data['Mem_SSD'] + data['MemType_2'] * data['Mem_2SSD'] )

data['Hybrid'] = ( data['MemType'] * data['Mem_Hybrid'] + data['MemType_2'] * data['Mem_2Hybrid'] )

data['Flash_Storage'] = ( data['MemType'] * data['Mem_Flash Storage'] + data['MemType_2'] * data['Mem_2Flash Storage'] )

# dropping unwanted columns

data.drop(columns=['Cpu','Memory','MemType','MemType_2','Mem_HDD','Mem_SSD','Mem_Hybrid','Mem_Flash Storage','Mem_2HDD','Mem_2SSD','Mem_2Hybrid','Mem_2Flash Storage'],inplace=True)

data.sample(5)

# GPU
data['Gpu'].value_counts()

# Handling GPU column: taking only brand names

data['Gpu'] = data['Gpu'].apply(lambda x:x.split()[0])
data.sample(10)

data['Gpu'].value_counts()

# Correlation between all features

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# taking copy of the dataframe
data_copy = data
data_copy.head()

data.isnull().sum()

"""**ENCODING**"""

# One-Hot Encoding for columns like Company, Type_Name, Gpu_BrandName, OS, CPU_name
'''
categorical_cols = ['Company','TypeName','Gpu','OpSys','CPU_name']

encoders = {}  # Dictionary to store encoders for each column
for col in categorical_cols:
    encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
    encoders[col] = encoder.fit_transform(data[[col]])

    # Encode and store
    data_encoded = pd.DataFrame(encoders[col], columns=encoder.get_feature_names_out([col]))
    data = pd.concat([data, data_encoded], axis=1)
    data.drop(col, axis=1, inplace=True)
'''

# Get dummies for Categorical columns

data = pd.get_dummies(data, columns=['Company','TypeName','Gpu','OpSys','CPU_name'], drop_first=True)

data.sample(5)

data.columns

data.shape

data.info()

# Determine error metrics / Linear regression

X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)


print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Adjusted r2:", adjusted_r2)

## method:2 - Linear Regression using Column Transformer Class and Pipeline

#test = np.log(data_copy['Price'])

test = data_copy['Price']
train = data_copy.drop(['Price'],axis = 1)

# Splitting data for Model Training/Testing
X_train, X_test, y_train, y_test = train_test_split(train,test,test_size=0.15,random_state=2)

# checking train/test data shape
X_train.shape,X_test.shape

# assigning index to train data columns & storing in a dictionary

mapper = {i:value for i,value in enumerate(X_train.columns)}
mapper

"""**Linear Regression**"""

# Using Column Transformer Class and Pipeline
# Column Transformer: Efficiently handles mixed-type data (numerical & categorical) by applying different transformations(OHE/Std.Scaler) to columns
# remainder parameter to define what to do with untransformed columns (e.g., pass through, apply a default transformer)

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,3,4,9])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',metrics.r2_score(y_test,y_pred))
print('MAE',metrics.mean_absolute_error(y_test,y_pred))

"""**Multiple Regressions**"""

# Passing data to Multiple Regression models

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Regression': SVR()
}

# Train and evaluate models
results = {}
for name, model in models.items():      ## key, value
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse,'R-squared': r2}

# results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"MAE: {result['MAE']}")
    print(f"MSE: {result['MSE']}")
    print(f"RMSE: {result['RMSE']}")
    print(f"R-squared: {result['R-squared']}")
    print()

# Visualize performance
scaler = MinMaxScaler()
metrics_df = pd.DataFrame(results).T    ## T-transposes the DataFrame, which swaps the rows and columns for visualization
metrics_df_scaled = pd.DataFrame(scaler.fit_transform(metrics_df), columns=metrics_df.columns, index=metrics_df.index)
metrics_df_scaled.plot(kind='bar', figsize=(10, 6),alpha=0.5)


plt.title('Performance Comparison of Regression Models')
plt.xlabel('Model')
plt.ylabel('Error Metrics')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

## Random Forest is found to be the best fit model with a Mean absolute error: 10864, and R-square score: 0.76
## Also, Gradient Boosting produces MAE: 11267 & R-squared: 0.76223605088801

"""**Hyperparameter tuning**"""

# Hyperparameter tuning using GridSearchCV
# evaluates all possible combinations of hyperparameter values from a predefined grid

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gbm = GradientBoostingRegressor(random_state=2)
grid_search = GridSearchCV(gbm, param_grid, cv=5, scoring='neg_mean_squared_error')   ##grid_search=model
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best GBM model
best_gbm = grid_search.best_estimator_

# Evaluate the best GBM model
y_pred_train = best_gbm.predict(X_train_scaled)
y_pred_test = best_gbm.predict(X_test_scaled)

train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R-squared Score:", train_r2)
print("Test R-squared Score:", test_r2)

## Best Hyperparameters: {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}

# Perform cross-validation (5 fold)

cv_scores = cross_val_score(gbm, X, y, cv=5, scoring='neg_mean_squared_error') ## entire X,y given for validation

# Convert negative MSE scores to positive values
cv_scores = -cv_scores

# cross-validation scores
print("Cross-Validation Scores (MSE):", cv_scores)
print("Average MSE:", cv_scores.mean())

"""**Questions to Explore:**

Which features have the most significant impact on laptop prices?

*   RAM, SSD, Screen Resolution, CPU

Can the model accurately predict the prices of laptops from lesser-known brands?

*   The model cannot predict the price of lesse known brands due to comparitively less data of such brands

Does the brand of the laptop significantly influence its price?

*   Yes, price range varies on top brands

How well does the model perform on laptops with high-end specifications compared to budget laptops?

*   As the number of budget laptops are comparitively more in the dataset, the model gets more data for training and which directly impacts the prediction.

What are the limitations and challenges in predicting laptop prices accurately?

*   ML models typically focus on technical specifications. Brand reputation and popularity can influence price more than technical features and it is usuallyincomprehensible for ML models. Also dataset size play a vital role in prediction accuracy.

How does the model perform when predicting the prices of newly released laptops not present in the training dataset?

*   If new features are present in upcoming laptop models the ML model may produce less accurate results


"""

# EOF
