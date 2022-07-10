
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns",None)
df = pd.read_csv('./weatherAUS.csv')
df.head()
df.shape
df.info()
def tmpClass(tmp):
    if tmp <= 18:
        return 0
    elif tmp >10 and tmp <=20:
        return 1
    elif tmp > 20 and tmp <=30:
        return 2
    elif tmp > 30:
        return 3
df['TempClass'] = df['Temp3pm'].apply(tmpClass)
print(df['TempClass'])

df['Date'] = pd.to_datetime(df['Date'])
df['Date'].dtype
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df.drop('Date', axis = 1, inplace = True)
df.head()

categorical_features = [column_name for column_name in df.columns if df[column_name].dtype == 'O']
print("Number of Categorical Features: {}".format(len(categorical_features)))
print("Categorical Features: ",categorical_features)

numerical_features = [column_name for column_name in df.columns if df[column_name].dtype != 'O']
print("Number of Numerical Features: {}".format(len(numerical_features)))
print("Numerical Features: ",numerical_features)
# `Checking for Null values:`
df[categorical_features].isnull().sum()
# list of categorical features which has null values:

categorical_features_with_null = [feature for feature in categorical_features if df[feature].isnull().sum()]
# Filling the missing(Null) categorical features with most frequent value(mode)`
# Filling the missing(Null) categorical features with most frequent value(mode)

for each_feature in categorical_features_with_null:
    mode_val = df[each_feature].mode()[0]
    df[each_feature].fillna(mode_val,inplace=True)
df[categorical_features].isnull().sum()
# checking null values in numerical features

df[numerical_features].isnull().sum()
plt.figure(figsize=(15,10))
sns.heatmap(df[numerical_features].isnull(),linecolor='white')
# visualizing the Null values in Numerical Features:

df[numerical_features].isnull().sum().sort_values(ascending = False).plot(kind = 'bar')
#checking for outliers using Box Plot:

for feature in numerical_features:
    plt.figure(figsize=(10,10))
    sns.boxplot(df[feature])
    plt.title(feature)

# checking for outliers using the statistical formulas:

df[numerical_features].describe()


features_with_outliers = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
# Replacing outliers using IQR:

for feature in features_with_outliers:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (IQR*1.5)
    upper_limit = q3 + (IQR*1.5)
    df.loc[df[feature]<lower_limit,feature] = lower_limit
    df.loc[df[feature]>upper_limit,feature] = upper_limit
for feature in numerical_features:
    plt.figure(figsize=(10,10))
    sns.boxplot(df[feature])
    plt.title(feature)

# list of numerical Features with Null values:

numerical_features_with_null = [feature for feature in numerical_features if df[feature].isnull().sum()]
numerical_features_with_null
# Filling null values uisng mean: 

for feature in numerical_features_with_null:
    mean_value = df[feature].mean()
    df[feature].fillna(mean_value,inplace=True)
df.isnull().sum()
df.head()

# Exploring RainTomorrow label


df['RainTomorrow'].value_counts().plot(kind='bar')
#Looks like Target variable is imbalanced. It has more 'No' values. If data is imbalanced, then it might decrease performance of model. As this data is released by the meteorological department of Australia, it doesn't make any sense when we try to balance target variable, because the truthfullness of data might descreases. So, let me keep it as it is.
#Exploring RainToday Variable:

sns.countplot(data=df, x="RainToday")
plt.grid(linewidth = 0.5)
plt.show()

plt.figure(figsize=(20,10))
ax = sns.countplot(x="Location", hue="RainTomorrow", data=df)
sns.lineplot(data=df,x='Sunshine',y='Rainfall',color='goldenrod')
num_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
df[num_features].hist(bins=10,figsize=(20,20))

categorical_features
# Encoding Categorical Features using replace function:

df['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)

# pd.get_dummies(df['RainToday'],drop_first = True)

df['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)
def encode_data(feature_name):
    
    ''' 
    
     function which takes feature name as a parameter and return mapping dictionary to replace(or map) categorical data 
     to numerical data.
     
    '''
    
    mapping_dict = {}
    unique_values = list(df[feature_name].unique())
    for idx in range(len(unique_values)):
        mapping_dict[unique_values[idx]] = idx
    print(mapping_dict)
    return mapping_dict

df['WindGustDir'].replace(encode_data('WindGustDir'),inplace = True)
df['WindDir9am'].replace(encode_data('WindDir9am'),inplace = True)
df['WindDir3pm'].replace(encode_data('WindDir3pm'),inplace = True)
df['Location'].replace(encode_data('Location'), inplace = True)
df.head()

# `Spliting data into input features and label`
X = df.drop(['RainTomorrow'],axis=1)
y = df['RainTomorrow']

# finding feature importance using ExtraTreesRegressor:

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
etr_model = ExtraTreesRegressor()
etr_model.fit(X,y)
etr_model.feature_importances_
# visualizing feature importance using bar graph:

feature_imp = pd.Series(etr_model.feature_importances_,index=X.columns)
feature_imp.nlargest(10).plot(kind='barh')
feature_imp
## `5) Split Data into Training and Testing Set`  <a class="anchor" id=""></a>
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
print("Length of Training Data: {}".format(len(X_train)))
print("Length of Testing Data: {}".format(len(X_test)))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)



start_time = time.time()
classifier_logreg = LogisticRegression(solver='liblinear', random_state=0)
classifier_logreg.fit(X_train, y_train)
end_time = time.time()
print("Time Taken to train: {}".format(end_time - start_time))
y_pred = classifier_logreg.predict(X_test)
print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred)))
print("Train Data Score: {}".format(classifier_logreg.score(X_train, y_train)))
print("Test Data Score: {}".format(classifier_logreg.score(X_test, y_test)))


print("Confusion Matrix:")
print("\n",confusion_matrix(y_test,y_pred))

print("classification_report:")
print("\n",classification_report(y_test,y_pred))


y_pred_logreg_proba = classifier_logreg.predict_proba(X_test)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_logreg_proba[:,1])
# Plotting ROC curve:

# plt.figure(figsize=(6,4))
# plt.plot(fpr,tpr,'-g',linewidth=1)
# plt.plot([0,1], [0,1], 'k--' )
# plt.title('ROC curve for Logistic Regression Model')
# plt.xlabel("False Positive Rate")
# plt.ylabel('True Positive Rate')
# plt.show()
# finding ROC-AUC score:

from sklearn.metrics import roc_auc_score
print('ROC AUC Scores: {}'.format(roc_auc_score(y_test, y_pred)))
# `Finding whether model performance can be improved using Cross Validation Score:`
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier_logreg, X_train, y_train, cv = 5, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
print('Average cross-validation score: {}'.format(scores.mean()))
# _`The mean accuracy score of cross validation is almost same like original model accuracy score which is 0.8445. So, accuracy of model may not be improved using Cross-validation.`_

from catboost import CatBoostClassifier
# `Model Training:`
start_time = time.time()
cat_classifier = CatBoostClassifier(iterations=2000, eval_metric = "AUC")
cat_classifier.fit(X_train, y_train)
end_time = time.time()
print("Time Taken to train: {}".format(end_time - start_time))
# `Model Testing:`
y_pred_cat = cat_classifier.predict(X_test)
print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred_cat)))
# `Checking for Overfitting and Under Fitting:`
print("Train Data Score: {}".format(cat_classifier.score(X_train, y_train)))
print("Test Data Score: {}".format(cat_classifier.score(X_test, y_test)))
# _`Accuracy Score of Training and Testing Data is comparable and almost equal. So, there is no question of Underfitting and Over Fitting. And model is generalizing well for new unseen data.`_
# Confusion Matrix:

print("Confusion Matrix:")
print("\n",confusion_matrix(y_test,y_pred_cat))
# classification Report:

print("classification_report:")
print("\n",classification_report(y_test,y_pred_cat))
# predicting the probabilities:

y_pred_cat_proba = cat_classifier.predict_proba(X_test)
# Finding True Positive Rate(tpr), False Positive Rate(fpr), threshold values to plot ROC curve  

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_cat_proba[:,1])
# plotting ROC Curve:

# plt.figure(figsize=(6,4))
# plt.plot(fpr,tpr,'-g',linewidth=1)
# plt.plot([0,1], [0,1], 'k--' )
# plt.title('ROC curve for Cat Boost Model')
# plt.xlabel("False Positive Rate")
# plt.ylabel('True Positive Rate')
# plt.show()
#finding ROC AUC Scores:

from sklearn.metrics import roc_auc_score
print('ROC AUC Scores: {}'.format(roc_auc_score(y_test, y_pred_cat)))
# `Finding whether model performance can be improved using Cross Validation Score:`
from sklearn.model_selection import cross_val_score
scores = cross_val_score(cat_classifier, X_train, y_train, cv = 5, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
print('Average cross-validation score: {}'.format(scores.mean()))
# _`The mean accuracy score of cross validation is almost same like original model accuracy score which is 0.8647050735597415. So, accuracy of model may not be improved using Cross-validation.`_

from sklearn.ensemble import RandomForestClassifier
# `Model Training:`
start_time = time.time()
classifier_rf=RandomForestClassifier()
classifier_rf.fit(X_train,y_train)
end_time = time.time()
print("Time Taken to train: {}".format(end_time - start_time))
# `Model Testing:`
y_pred_rf = classifier_rf.predict(X_test)
print("Accuracy Score: {}".format(accuracy_score(y_test,y_pred_rf)))
# `Checking for Overfitting and Under Fitting:`
print("Train Data Score: {}".format(classifier_rf.score(X_train, y_train)))
print("Test Data Score: {}".format(classifier_rf.score(X_test, y_test)))
# _`Accuracy score for Training Set is almost 1 or 100%, which is quite uncommon. And testing accuracy is 0.85. It seems like model is overfitting, because the generalization for unseen data is not that accurate, when compared with seen data and difference between training - testing accuracy is not minimum.`_
## `8) Results and Conclusion:`  <a class="anchor" id=""></a>
# `Best Models in terms of accuracy (In my Experiment):`

#     1) Cat Boost Model
#     2) Logistic Regression
#     3) Random Forest
    
# `Best Models in terms of Computation Time (In my Experiment):`

#     1) Logistic Regression
#     2) Random Forest
#     3) Cat Boost Model
        
# `Conclusion:`

# The accuracy score of Cat Boost Model is high when compared with accuracy scores of Logistic Regression and Random Forest. But cat Boost model consumes lot of time to train the model.

# In terms of computation time and Accuracy score, logistic Regression model is doing job.

## `9) Saving Classifier Object into Pickle File:`  <a class="anchor" id="10"></a>
with open('logreg.pkl', 'wb') as file:
    pickle.dump(classifier_logreg, file)
with open('catboostclassifier.pkl', 'wb') as file:
    pickle.dump(cat_classifier, file)

