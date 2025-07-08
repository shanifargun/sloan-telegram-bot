# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# + [markdown] id="Lf5tfCiyQNRh"
# This notebook is derived from source code and notebooks from the open source repository at https://github.com/Kaggle/learntools and modified for classroom use in SCM.256 with supply chain management examples.
#
# ```
# Copyright (2021) Dan Becker
# Copyright (2023) Elenna Dugundji & Thomas Koch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ```

# + [markdown] id="C_4ZHN9k87q1"
# # Big Mart Item Sales Prediction per SKU per Store

# + [markdown] id="bhxDBu5BCJFR"
# # Explainable AI (XAI)
#
# Many people say machine learning models are "black boxes", in the sense that they can make good predictions but you can't understand the logic behind those predictions.
# However, this week you will learn techniques to extract the following insights from sophisticated machine learning models.
#
# - What features in the data did the model think are most important?
# - How does each feature affect the model's predictions in a big-picture sense: what is its typical effect when considered over a large number of possible predictions?

# + [markdown] id="pY9SinB4DCuy"
# ## Why Are Feature Insights Valuable?
#
# Feature insights have many uses, including
# - Debugging
# - Informing feature engineering
# - Directing future data collection
# - Informing human decision-making
# - Building Trust
#
#
# ### Debugging
# The world has a lot of unreliable, disorganized and generally dirty data. You add a potential source of errors as you write preprocessing code. Add in the potential for target leakage, and it is the norm rather than the exception to have errors at some point in a real data science project.
#
# Given the frequency and potentially disastrous consequences of bugs, debugging is one of the most valuable skills in data science. Understanding the patterns a model is finding will help you identify when those are at odds with your knowledge of the real world, and this is typically the first step in tracking down bugs.
#
# ### Informing Feature Engineering
# Feature engineering is usually the most effective way to improve model accuracy. Feature engineering usually involves repeatedly creating new features using transformations of your raw data or features you have previously created.
#
# Sometimes you can go through this process using nothing but intuition about the underlying topic. But you'll need more direction when you have 100s of raw features or when you lack background knowledge about the topic you are working on. As an increasing number of datasets start with 100s or 1000s of raw features, this approach is becoming increasingly important.
#
# ### Directing Future Data Collection
# You have no control over datasets you download online. But many businesses and organizations using data science have opportunities to expand what types of data they collect. Collecting new types of data can be expensive or inconvenient, so they only want to do this if they know it will be worthwhile. Model-based insights give you a good understanding of the value of features you currently have, which will help you reason about what new values may be most helpful.
#
# ### Informing Human Decision-Making
# Some decisions are made automatically by models. Amazon doesn't have humans (or elves) scurry to decide what to show you whenever you go to their website.  But many important decisions are made by humans. For these decisions, insights can be more valuable than predictions.
#
# ### Building Trust
# Many people won't assume they can trust your model for important decisions without verifying some basic facts. This is a smart precaution given the frequency of data errors. In practice, showing insights that fit their general understanding of the problem will help build trust, even among people with little deep knowledge of data science.

# + [markdown] id="wHdFlRdAE2F3"
# ## Setting Up the Workspace

# + id="Iq3XqCQ2Ez-R"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import shap  # package used for Shapley additive explanations

# + [markdown] id="5a934439"
# ## Loading the Data

# + [markdown] id="9ZLic5nQzBYK"
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and predict the sales of each product at a particular outlet.
#
# Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.
#
# Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.
#
#
# ### Features
#
# |        Column Name        |                                             Description                                             |
# |:-------------------------:|:---------------------------------------------------------------------------------------------------:|
# | Item_Identifier           | Unique product ID                                                                                   |
# | Item_Weight               | Weight of product                                                                                   |
# | Item_Fat_Content          | Whether the product is low fat or regular                                                           |
# | Item_Visibility           | The percentage of total display area of all products in a store allocated to the particular product |
# | Item_Type                 | The category to which the product belongs                                                           |
# | Item_MRP                  | Maximum Retail Price (list price) of the product (in rupees)                                        |
# | Outlet_Identifier         | Unique store ID                                                                                     |
# | Outlet_Establishment_Year | The year in which store was established                                                             |
# | Outlet_Size               | The size of the store in terms of ground area covered                                               |
# | Outlet_Location_Type      | The type of area in which the store is located                                                      |
# | Outlet_Type               | Whether the outlet is a grocery store or some sort of supermarket                                   |
# | Item_Outlet_Sales         | Sales of the product in the particular store. This is the target variable to be predicted.          |
#

# + colab={"base_uri": "https://localhost:8080/", "height": 313} id="28a8a269" outputId="f4a8b1b2-08cf-4f63-bc2a-85d594382c54"
df_sales = pd.read_csv('https://www.dropbox.com/s/yqaymhdf7bvvair/bigmart_sales_predictions.csv?dl=1')
df_sales.head(5)

# + [markdown] id="5701c672"
# Remove columns we will not use today

# + id="a198294b"
df_sales = df_sales.drop(columns=['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'])

# + [markdown] id="34f2534f"
# Check for potential duplicated records

# + colab={"base_uri": "https://localhost:8080/"} id="00d3e2fd" outputId="3a29d447-1cad-4b58-dbd6-cf7a573264b6"
df_sales.duplicated().sum()

# + [markdown] id="16eec147"
# There are no duplicated records, but are there missing values?

# + colab={"base_uri": "https://localhost:8080/", "height": 366} id="953cf1ed" outputId="6526260d-be89-44d2-d259-5566b7a19775"
df_sales.isna().sum()

# + [markdown] id="c303a2fb"
# There are missing values!
# Remember the pipeline we showed you in class before?
#
# We will use that to impute the values for `item_weight` and `Outlet_Size`. As we can see in the `value_counts()` below, `Outlet_Size` is an excellent candidate for an ordinal encoding.

# + colab={"base_uri": "https://localhost:8080/", "height": 209} id="b8a62e20" outputId="3dcf83fe-ede3-4b1b-f598-8b721735320b"
df_sales['Outlet_Size'].value_counts()

# + [markdown] id="39d05a3c"
# To clean up the column `Item_Fat_Content` we manually consolidate the 5 values into two values in an ordinal way.

# + colab={"base_uri": "https://localhost:8080/", "height": 272} id="80dfb7b0" outputId="c835f3a2-7527-47ca-91fc-1c1e1a42e368"
df_sales['Item_Fat_Content'].value_counts()

# + colab={"base_uri": "https://localhost:8080/", "height": 178} id="cf57ae0f" outputId="7e48c445-8807-49d2-b0bf-636d899a79a3"
df_sales['Item_Fat_Content'] = df_sales['Item_Fat_Content'].replace(
    {'Low Fat': 'low',
     'LF': 'low',
     'low fat': 'low',
     'Regular': 'regular',
     'reg': 'regular'})
df_sales['Item_Fat_Content'].value_counts()

# + colab={"base_uri": "https://localhost:8080/", "height": 366} id="bfe99a8e" outputId="f6b66768-6c21-4f9e-eeae-0bf4e41275e9"
df_sales.dtypes

# + [markdown] id="AvbYpMWl3P1G"
# ### Visualizations

# + colab={"base_uri": "https://localhost:8080/", "height": 472} id="vAwxd_TM3xRy" outputId="788f265e-2090-427f-e773-79e13d49cd0a"
# Create histograms to view distributions of various features in the dataset
ax = sns.histplot(data = df_sales, x = 'Item_Outlet_Sales')
median = df_sales['Item_Outlet_Sales'].median()
ax.set(title = 'Distribution of Item Sales')
ax.axvline(median, color = 'black', linestyle = '--',
            label = f'Median Item Sales = ${median}')
ax.legend();

# + colab={"base_uri": "https://localhost:8080/", "height": 472} id="ea_WkjgE36Cf" outputId="884b0dfc-78ec-4434-d7cf-e71d077a2c1f"
# Create histograms to view distributions of various features in the dataset
ax = sns.histplot(data = df_sales, x = 'Item_MRP')
median = df_sales['Item_MRP'].median()
ax.set(title = 'Distribution of Item MRP')
ax.axvline(median, color = 'black', linestyle = '--',
            label = f'Median Item MRP = ${median}')
ax.legend();

# + colab={"base_uri": "https://localhost:8080/", "height": 485} id="-ScBfUDg4H2c" outputId="2be5019f-3069-448f-8e26-8217c98e9ac2"
# Create boxplots to view statistical summaries of Outlet_Size and Item_Outlet_Sales
sns.boxplot(x = 'Outlet_Size', y = 'Item_Outlet_Sales', data = df_sales)
plt.xticks(rotation = 45);

# + colab={"base_uri": "https://localhost:8080/", "height": 473} id="yTpGT8Qk3PGr" outputId="e85b584e-39f2-4088-f075-d8ad4db7e3fb"
# Create boxplots to view statistical summaries of Outlet_Location_Type and Item_Outlet_Sales
sns.boxplot(x = 'Outlet_Location_Type', y = 'Item_Outlet_Sales', data = df_sales)
plt.xticks(rotation = 45);

# + colab={"base_uri": "https://localhost:8080/", "height": 585} id="rwmV9q-u3-VT" outputId="f9fed1b4-3282-4f6a-a074-54370e15fbf6"
# Create violinplots to view statistical summaries of Item_Type and Item_Outlet_Sales
sns.violinplot(x = 'Item_Type', y = 'Item_Outlet_Sales', data = df_sales)
plt.xticks(rotation = 90);

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="206Z3rnj3X_H" outputId="d71b61c7-3f40-4d74-d173-4a263418223b"
# Display correlation between numeric features
corr = df_sales.select_dtypes(include=['number']).corr()
sns.heatmap(corr, cmap = 'Greens', annot = True);

# + colab={"base_uri": "https://localhost:8080/", "height": 472} id="YwtZh2123iim" outputId="37240030-b17b-44fa-f9ae-07ad2f76c9f4"
# Construct scatterplot of Item_MRP and Sales
ax = sns.scatterplot(data = df_sales, x = 'Item_MRP', y = 'Item_Outlet_Sales', hue = 'Outlet_Type', palette = 'plasma')
ax.set_title('Item MRP vs. Item Outlet Sales')
ax.set_xlabel('Item MRP')
ax.set_ylabel('Item Outlet Sales')
ax.legend(bbox_to_anchor = [1,1]);

# + [markdown] id="59c9a931"
# ### Pipeline

# + colab={"base_uri": "https://localhost:8080/", "height": 188} id="79d788fb" outputId="9f2d2560-8868-40e7-835e-acd556b879f3"
#from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
#from sklearn.impute import SimpleImputer
#from sklearn.compose import make_column_transformer, make_column_selector
#from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import train_test_split
#from sklearn import set_config
#from sklearn.pipeline import Pipeline

#Establish each pipeline for differente feature types
categorical_features = ['Item_Type','Outlet_Location_Type','Outlet_Type']
cat_pipe = Pipeline(steps=[
    ('simpleimputer', SimpleImputer(strategy='most_frequent')), #Impute missing by using the most frequent value along each column
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) #Encode as one-hot

categorical_ordinal_features = ['Outlet_Size','Item_Fat_Content']
categorical_ordinal_values = [
    ['Missing','Small','Medium','High'], #Values for Outlet_Size
    ['Missing', 'low', 'regular'] #Values for Item_Fat_Content
]
cat_ord_pipe = Pipeline(steps=[
    ('simpleimputer', SimpleImputer(strategy='constant', fill_value='Missing')), #Impute missing values by setting them as 'Missing'
    ('encoder', OrdinalEncoder(handle_unknown='error', #We should not see unknown values, as they are all imputed in the pipeline
                                     categories=categorical_ordinal_values))]) #Encode as ordinal, in order as given.

num_features = ['Item_Weight','Item_MRP','Item_Visibility']
num_pipe = Pipeline(steps=[
    ('simpleimputer', SimpleImputer()), #Impute missing values by setting those to the mean value along each column
    ('standardscaler', StandardScaler())]) #Standardize numerical features by removing the mean and scaling to unit variance

#Create rail switch to correctly route each column to the correct pipel
preprocessor = make_column_transformer(
    (cat_pipe, categorical_features),
    (cat_ord_pipe, categorical_ordinal_features),
    (num_pipe, num_features), verbose_feature_names_out=False).set_output(transform="pandas")
preprocessor

# + [markdown] id="48f590ba"
# Establish our X and Y

# + colab={"base_uri": "https://localhost:8080/", "height": 143} id="f56f250a" outputId="10b437a2-949f-4bca-9720-164f1c0c499b"
target = "Item_Outlet_Sales"
y = df_sales[target] #Take series with target values from dataframe.
X = df_sales.drop(columns=[target]) #Remove target series from dataframe with features.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #Split into 80-20 train-test set, data IS shuffled before splitting
X_train.head(3)

# + [markdown] id="3a8188a7"
# Use our preprocessing pipeline to pre-process our data and save the result to a dataframe.

# + id="KN4YcJYb-acs"


# + colab={"base_uri": "https://localhost:8080/", "height": 461} id="ba4c2db9" outputId="86df8aed-ad69-47fe-9db3-fc6c9a1a9c24"
#We first need to fit each step in our pipeline to the training data.
preprocessor.fit(X_train)

#Now we can transform our data, with the steps specified in the pipeline
#The output of transform() our pipeline is a numpy array.
# We convert this data back into a pandas dataframe using the list of feature names stored in our preprocessor.

Xprocessed_train_df = preprocessor.transform(X_train)
Xprocessed_test_df = preprocessor.transform(X_test)

#Note the changed order of the index caused by the shuffling in train_test_split!
display(Xprocessed_train_df)

# + [markdown] id="5EpUs9KjLHMF"
# ---
# # 1. Feature Permutation Importance
#
# One of the most basic questions we might ask of a model is: What features have the biggest impact on predictions?  
#
# This concept is called **feature importance**.
#
# There are multiple ways to measure feature importance.  Some approaches answer subtly different versions of the question above. Other approaches have documented shortcomings.
#
# In this lesson, we'll focus on **permutation importance**.  Compared to most other approaches, permutation importance is:
#
# - fast to calculate,
# - widely used and understood, and
# - consistent with properties we would want a feature importance measure to have.

# + [markdown] id="7325daf3" papermill={"duration": 0.00944, "end_time": "2021-11-09T00:04:42.388349", "exception": false, "start_time": "2021-11-09T00:04:42.378909", "status": "completed"}
# ## How It Works
#
# Permutation importance uses models differently than anything you've seen so far, and many people find it confusing at first. So we'll start with an example to make it more concrete.  
#
# Consider data with the following format:
#
# | Item Visibility | Max Retail Price | ... | Item Outlet Sales |
# |-----------------|------------------|-----|-------------------|
# | 1               | 300              | ... | 5000              |
# | 0.5             | 1                | ... | 1000              |
# | ...             | ...              | ... | ...               |
# | 0.5             | 2.5              | ... | 2500              |
# | 1               | 3                | ... | 1800              |
#
# We want to predict the sales of a SKU at a store, based on features such as the visiblity of the item in the store and the maximum retail price of that SKU, as well as some other features.
#
# **Permutation importance is calculated after a model has been fitted.** So we won't change the model or change what predictions we'd get for a given value of the features.
#
# Instead we will ask the following question:  If I randomly shuffle a single column of the validation data, leaving the target and all other columns in place, how would that affect the accuracy of predictions in that now-shuffled data?
#
# Randomly re-ordering a single column should cause less accurate predictions, since the resulting data no longer corresponds to anything observed in the real world.  Model accuracy especially suffers if we shuffle a column that the model relied on heavily for predictions.  In this case, shuffling `Max Retail Price` would cause terrible predictions. If we shuffled `Item Visibility` instead, the resulting predictions wouldn't suffer nearly as much.
#
#
# ![Imgur](https://i.imgur.com/7EcnpWe.png)
#
#
# With this insight, the process is as follows:
#
# 1. Get a trained model.
# 2. Shuffle the values in a single column, make predictions using the resulting dataset.  Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
# 3. Return the data to the original order (undoing the shuffle from step 2). Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.
#
#

# + [markdown] id="126b7e46"
# ## Code Example
#
# Our example will use a model that predicts the sale of one SKU at a store based on the features of that item and store. Model-building isn't our current focus, so the cell below loads the data and builds a rudimentary model.

# + [markdown] id="eafd49db" papermill={"duration": 0.004721, "end_time": "2021-11-09T00:04:44.009580", "exception": false, "start_time": "2021-11-09T00:04:44.004859", "status": "completed"}
# Here is how to calculate and show importances with the scikit library:

# + colab={"base_uri": "https://localhost:8080/", "height": 927} id="2781876b" outputId="5521aa84-9773-470c-c9a7-745a0485796a" papermill={"duration": 8.438742, "end_time": "2021-11-09T00:04:52.453249", "exception": false, "start_time": "2021-11-09T00:04:44.014507", "status": "completed"}
from sklearn.inspection import permutation_importance
#from sklearn.ensemble import RandomForestRegressor

my_model = RandomForestRegressor(n_estimators=100,
                                  random_state=0).fit(Xprocessed_train_df, y_train)

# Compute permutation importance
perm = permutation_importance(my_model, Xprocessed_test_df, y_test, n_repeats=10, random_state=1) #, scoring='neg_mean_absolute_error')

feature_importance = pd.DataFrame(perm.importances_mean, index=Xprocessed_test_df.columns, columns=['perm_importances_mean'])
feature_importance = pd.DataFrame({
    'perm_importances_mean': perm.importances_mean,
    'perm_importances_std': perm.importances_std  # Adding standard deviation
},index=Xprocessed_test_df.columns)

# Display results
feature_importance = feature_importance.sort_values(by='perm_importances_mean',ascending=False)
feature_importance

# + colab={"base_uri": "https://localhost:8080/", "height": 581} id="gzVkhb71Rp-2" outputId="6434c857-2a9f-4934-b38c-55d85fbe26c6"
# Create bar chart
plt.figure(figsize=(10, 6))
ax = feature_importance.perm_importances_mean.plot.barh()
plt.xlabel("Permutation Importance (Mean Decrease in Score)")
plt.ylabel("Features")
plt.title("Feature Importance (Permutation Method)")

# + [markdown] id="53b0238f" papermill={"duration": 0.006388, "end_time": "2021-11-09T00:04:52.468724", "exception": false, "start_time": "2021-11-09T00:04:52.462336", "status": "completed"}
# ## Interpreting Permutation Importances
#
# The values towards the top are the most important features, and those towards the bottom matter least.
#
# The first number in each row shows how much model performance decreased with a random shuffling (in this case, using "accuracy" as the performance metric).
#
# Like most things in data science, there is some randomness to the exact performance change from a shuffling a column.  We measure the amount of randomness in our permutation importance calculation by repeating the process with multiple shuffles.  The number after the **±** measures how performance varied from one-reshuffling to the next.
#
# You'll occasionally see negative values for permutation importances. In those cases, the predictions on the shuffled (or noisy) data happened to be more accurate than the real data. This happens when the feature didn't matter (should have had an importance close to 0), but random chance caused the predictions on shuffled data to be more accurate. This is more common with small datasets, like the one in this example, because there is more room for luck/chance.
#
# In our example, the most important feature was **Item_MRP**, the maximum retail price of that SKU. That seems sensible.

# + [markdown] id="1OrA1-haM9nb"
# ---
# # 2. SHAP values (SHapley Additive exPlanations)
#
# You've seen (and used) Feature permutation to extract general insights from a machine learning model. But what if you want to break down how the model works for an **individual prediction**?
#
# SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the impact of each feature.
#
#

# + [markdown] id="68786496" papermill={"duration": 0.008052, "end_time": "2021-11-09T00:06:44.382959", "exception": false, "start_time": "2021-11-09T00:06:44.374907", "status": "completed"}
# ## How It Works
# SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.
#
# In this notebook, we predicted the sales of different SKU's at different stores.
#
# We could ask:
# - How much was a prediction driven by the fact that the maximum retail price of a product was 200 rupees?
#     
# But it's easier to give a concrete, numeric answer if we restate this as:
# - How much was a prediction driven by the fact that the maximum retail price of a product was 200 rupees, **instead of some baseline number of the maximum retail price.**
#
# Of course, each store and item, have many features. So if we answer this question for `maximum retail price`, we could repeat the process for all other features.
#
# SHAP values do this in a way that guarantees a nice property. Specifically, you decompose a prediction with the following equation:
#
# ```sum(SHAP values for all features) = pred_for_sku_and_store - pred_for_baseline_values```
#
# That is, the SHAP values of all features sum up to explain why our prediction was different from the baseline. This allows us to decompose a prediction in a graph like this:
#
# ![Imgur](https://i.ibb.co/GvtpDf6/Screenshot-from-2023-03-13-07-18-26.png)

# + [markdown] id="8114e5f0" papermill={"duration": 0.006602, "end_time": "2021-11-09T00:06:44.396809", "exception": false, "start_time": "2021-11-09T00:06:44.390207", "status": "completed"}
#
# How do you interpret this?
#
# We predicted a total sales of 3638.77 rupees for this SKU at a store, whereas the base_value is 2179.  Feature values causing increased predictions are in red, and their visual size shows the magnitude of the feature's effect.  Feature values decreasing the prediction are in blue.  The biggest impact comes from `Item_MRP` being 1.371.  Though the store not being a Type 3 supermarket and the lower `Item_Visbility` have a meaningful effect decreasing the prediction.
#
# If you subtract the length of the blue bars from the length of the pink bars, it equals the distance from the base value to the output.
#
# There is some complexity to the technique, to ensure that the baseline plus the sum of individual effects adds up to the prediction (which isn't as straightforward as it sounds). We won't go into that detail here, since it isn't critical for using the technique. [This blog post](https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d) has a longer theoretical explanation.
#
# ## Code Example
# We calculate SHAP values using the wonderful [Shap](https://github.com/slundberg/shap) library.
#
# For this example, we'll reuse the model you've already seen with the Big Mart data.

# + id="fc15040e" papermill={"duration": 1.551649, "end_time": "2021-11-09T00:06:45.955494", "exception": false, "start_time": "2021-11-09T00:06:44.403845", "status": "completed"}
#my_model = RandomForestRegressor(n_estimators=100, random_state=0).fit(Xprocessed_train_df, y_train)

#import xgboost as xgb

my_model = xgb.XGBRegressor(random_state=0).fit(Xprocessed_train_df, y_train)
# -

# This code calculates feature importance using the .get_fscore() method from an XGBoost model's booster.

# + colab={"base_uri": "https://localhost:8080/", "height": 533} id="rc0NTo0BrFzZ" outputId="f3f9c5ef-eb51-477e-8ff4-d36fe03b7a1b"
feature_important = my_model.get_booster().get_fscore()
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["f-score"]).sort_values(by = "f-score", ascending=False)
data.nlargest(25, columns="f-score").plot(kind='barh', figsize = (20,10)) ## plot top 25 features

# + [markdown] id="e1feb9fc" papermill={"duration": 0.006494, "end_time": "2021-11-09T00:06:45.971254", "exception": false, "start_time": "2021-11-09T00:06:45.964760", "status": "completed"}
# We will look at SHAP values for a single row of the dataset (we arbitrarily chose index 5223). For context, we'll look at the raw predictions before looking at the SHAP values.

# + [markdown] id="7f6c29f5" papermill={"duration": 0.006853, "end_time": "2021-11-09T00:06:46.021120", "exception": false, "start_time": "2021-11-09T00:06:46.014267", "status": "completed"}
# The store and SKU have a predicted total sales of 2629 rupees.
#
# Now, we'll move onto the code to get SHAP values for that single prediction.

# + colab={"base_uri": "https://localhost:8080/", "height": 43} id="2768c8f1" outputId="2d4bcef1-209b-4d11-d5f7-1a3f5eaad48e" papermill={"duration": 1.544191, "end_time": "2021-11-09T00:06:47.572466", "exception": false, "start_time": "2021-11-09T00:06:46.028275", "status": "completed"}
import shap  # package used to calculate SHAP values

#Initializes the JavaScript visualization environment used by SHAP in Jupyter notebooks or IPython environments.
shap.initjs()

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_explainer_values = explainer(Xprocessed_train_df, y_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 617} id="f7mgWzp1eQO6" outputId="d6429056-f24b-455d-f5de-7b8afbc58ced"
shap.waterfall_plot(shap_explainer_values[5223])

# + colab={"base_uri": "https://localhost:8080/", "height": 193} id="796709c0" outputId="cf6afa41-ec2d-4677-81d1-1f42dd667854" papermill={"duration": 0.045472, "end_time": "2021-11-09T00:06:47.642462", "exception": false, "start_time": "2021-11-09T00:06:47.596990", "status": "completed"}
shap.initjs()
shap.force_plot(shap_explainer_values[5223])

# + [markdown] id="KlF67SNSx5gH"
# According to this force plot, the base value for sales is 2179 rupees.
# The store sales for this particular example is 2629 rupees.
# The most influential factors that contribute to increasing the sales are the Maximum Retail Price and the store outlet not being a Grocery Store. The outlet type not being Supermarket Type 3 is the biggest factor reducing the sales estimate.

# + [markdown] id="83114595" papermill={"duration": 0.019936, "end_time": "2021-11-09T00:06:47.683260", "exception": false, "start_time": "2021-11-09T00:06:47.663324", "status": "completed"}
# If you look carefully at the code where we created the SHAP values, you'll notice we reference Trees in `shap.TreeExplainer(my_model)`.  But the SHAP package has explainers for every type of model.
#
# - `shap.DeepExplainer` works with Deep Learning models.
# - `shap.KernelExplainer` works with all models, though it is slower than other Explainers and it offers an approximation rather than exact Shap values.
#
#

# + [markdown] id="9bfd13f0" papermill={"duration": 0.008891, "end_time": "2021-11-09T00:04:34.799990", "exception": false, "start_time": "2021-11-09T00:04:34.791099", "status": "completed"}
# ---
# # 3. Advanced Uses of SHAP Values
#
# Shap values show how much a given feature changed our prediction (compared to if we made that prediction at some baseline value of that feature).
#
# For example, consider an ultra-simple model:
#     $$y = 4 * x1 + 2 * x2$$
#
# If $x1$ takes the value 2, instead of a baseline value of 0, then our SHAP value for $x1$ would be 8 (from 4 times 2).
#
# These are harder to calculate with the sophisticated models we use in practice. But through some algorithmic cleverness, Shap values allow us to decompose any prediction into the sum of effects of each feature value, yielding a graph like this:
#
# ![Imgur](https://i.ibb.co/GvtpDf6/Screenshot-from-2023-03-13-07-18-26.png)
#
# In addition to this nice breakdown for each prediction, the [Shap library](https://github.com/slundberg/shap) offers great visualizations of groups of Shap values. We will focus on two of these visualizations. These visualizations have conceptual similarities to permutation importance. So threads from the previous exercises will come together here.

# + [markdown] id="75ab9b24" papermill={"duration": 0.00682, "end_time": "2021-11-09T00:04:36.343711", "exception": false, "start_time": "2021-11-09T00:04:36.336891", "status": "completed"}
#
# ## SHAP Summary Plots
#
# Permutation importance is great because it created simple numeric measures to see which features mattered to a model. This helped us make comparisons between features easily, and you can present the resulting graphs to non-technical audiences.
#
# But it doesn't tell you how each features matter. If a feature has medium permutation importance, that could mean it has
# - a large effect for a few predictions, but no effect in general, or
# - a medium effect for all predictions.
#
# SHAP summary plots give us a birds-eye view of feature importance and what is driving it. We'll walk through an example plot for the Bigmart data:

# + id="d7d84eb9" papermill={"duration": 2.06839, "end_time": "2021-11-09T00:04:38.421323", "exception": false, "start_time": "2021-11-09T00:04:36.352933", "status": "completed"}
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate shap values now for Xprocessed_test_df. This is what we will plot.
shap_values = explainer.shap_values(Xprocessed_test_df)

# + colab={"base_uri": "https://localhost:8080/", "height": 983} id="YJZ1vYChoWDf" outputId="6b4d9bb1-17e4-4cc6-8b72-f983789e0e4f"
#shap.initjs()

# Display a bar plot of shap_values with a summary of all data in Xprocessed_test_df (instead of a single row)
shap.summary_plot(shap_values, Xprocessed_test_df, plot_type='bar')

# + [markdown] id="E57VN7VipAlx"
# Our shap summary plot we see that the four most impactful features are:
#
# * `Item_MRP`: Maximum Retail Price (list price) of the product (in rupees)
# *  Whether the store is a grocery store, whether the store is a Type 3
# *  Super market
# *  The item visiblity: percentage of the total display area of all products in the store allocated to that particular product

# + colab={"base_uri": "https://localhost:8080/", "height": 957} id="X78GWyroof-l" outputId="7a560cad-361f-4393-bf9b-b57d07e685f9"
# Display a summary plot of shap_values for all data in Xprocessed_test_df colored by value of feature
shap.summary_plot(shap_values, Xprocessed_test_df)

# + [markdown] id="Ry_yPRBymijo"
# The next plot is made of many dots, each dot has three characteristics:
# - Vertical location shows what feature it is depicting
# - Color shows whether that feature was high or low for that row of the dataset
# - Horizontal location shows whether the effect of that value caused a higher or lower prediction.
#
# For example, the point in the upper right was for a item and SKU that sold very well, increasing the prediction by 4000.
#
# If you look for long enough, there's a lot of information in this graph:
#
# According to the dot plot the `Item_MRP` is the most impactful feature in our model and very highly correlated with sales.
# The higher the MRP, the higher the target value, which was the outlet sales.
# If `Item_MRP` decreases, so does outlet sales
#
# The second most impactful feature is whether the store is a grocery store or not. Grocery stores have a dramatic negative impact on the target value.
#
# The third most impactful feature is the one-hot feature of Supermarket Type3.
# Supermarkets (and especially those that are type3) produce much higher sales than grocery stores.
#
# Finally we see that the impact of the item visiblity is a mixed bag with regard to positive or negative impact.

# + [markdown] id="jiGklrvEZX0I"
# The code we have shown isn't too complex. But there are a few caveats.
#
# - When plotting, we call `shap_values`.  For classification problems, there is a separate array of SHAP values for each possible outcome. In that case, we index in to get the SHAP values for the prediction of each class.
# - Calculating SHAP values can be slow. It isn't a problem here, because this dataset is small.  But you'll want to be careful when running these to plot with reasonably sized datasets.  The exception is when using an `xgboost` model, which SHAP has some optimizations for and which is thus much faster.
#
# This provides a great overview of the model, but we might want to delve into a single feature. That's where SHAP dependence contribution plots come into play.
#

# + [markdown] id="515f71d7" papermill={"duration": 0.008848, "end_time": "2021-11-09T00:04:38.440041", "exception": false, "start_time": "2021-11-09T00:04:38.431193", "status": "completed"}
# ## SHAP Dependence Contribution Plots
#
# What is the distribution of effects? Is the effect of having a certain value pretty constant, or does it vary a lot depending on the values of other feaures. SHAP dependence contribution plots add more detail.
#
# We get the dependence contribution plot with the following code. The only line that's different from the `summary_plot` is the last line.

# + colab={"base_uri": "https://localhost:8080/", "height": 502} id="e538443a" outputId="90ab12d7-7d1f-4367-e0f0-a097dd18a459" papermill={"duration": 0.338553, "end_time": "2021-11-09T00:04:38.787656", "exception": false, "start_time": "2021-11-09T00:04:38.449103", "status": "completed"}
shap.initjs()

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate SHAP values for Xprocessed_test_df. This is what we will plot.
shap_values = explainer.shap_values(Xprocessed_test_df)

# Make a dependence plot for 'Item_MRP' colored by a feature that SHAP chooses
shap.dependence_plot('Item_MRP', shap_values, Xprocessed_test_df)

# + [markdown] id="aXbvxQY2T-_T"
# Start by focusing on the shape, and we'll come back to color in a minute.  Each dot represents a row of the data. The horizontal location is the actual value from the dataset, and the vertical location shows what having that value did to the prediction.  The fact this slopes upward says that the higher the Item Max Retail Price, the higher the model's prediction is for Item Sales.
#
# The spread suggests that other features must interact with Item Max Retail Price.  
#
# For comparison, a simple linear regression would produce plots that are perfect lines, without this spread.
#
# This suggests we delve into the interactions, and the plots include color coding to help do that. While the primary trend is upward, you can visually inspect whether that varies by dot color.
#
#

# + [markdown] id="34ddf2e9" papermill={"duration": 0.010666, "end_time": "2021-11-09T00:04:38.809197", "exception": false, "start_time": "2021-11-09T00:04:38.798531", "status": "completed"}
# If you don't supply an argument for `interaction_index`, Shapley uses some logic to pick one that may be interesting.
#
# This didn't require writing a lot of code. But the trick with these techniques is in thinking critically about the results rather than writing code itself.
