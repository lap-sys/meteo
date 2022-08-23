
import sys
import streamlit as st
# import pandas as pd
# import numpy as np
# import time
# import math
import pickle
# from sklearn.model_selection import train_test_split
# from neuralprophet import NeuralProphet
# from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# import statsmodels.api as sm
# from sklearn.preprocessing import StandardScaler, normalize
# from imblearn.over_sampling import RandomOverSampler, SMOTE
st.write("Météo Forecast")
region =  st.selectbox(
    'Select an Australian city',
 ('Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany','Witchcliffe', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole','Hobart', 'Launceston','AliceSprings','Darwin','Katherine', 'Uluru'))

# dff = pd.read_csv('./weatherAUS.csv')
# df = dff
# df['Date'] = pd.to_datetime(df['Date'])
# df['year'] = df['Date'].dt.year
# df['month'] = df['Date'].dt.month
# df['day'] = df['Date'].dt.day
# df.drop('Date', axis = 1, inplace = True)


# categorical_features = [column_name for column_name in df.columns if df[column_name].dtype == 'O']

# numerical_features = [column_name for column_name in df.columns if df[column_name].dtype != 'O']

# categorical_features_with_null = [feature for feature in categorical_features if df[feature].isnull().sum()]

# for each_feature in categorical_features_with_null:
#     mode_val = df[each_feature].mode()[0]
#     df[each_feature].fillna(mode_val,inplace=True)

# print(df[numerical_features].describe())

# features_with_outliers = ['MaxTemp', 'Rainfall', 'WindGustSpeed', 'Humidity9am', 'Pressure3pm']

# for feature in features_with_outliers:
#     q1 = df[feature].quantile(0.25)
#     q3 = df[feature].quantile(0.75)
#     IQR = q3-q1
#     lower_limit = q1 - (IQR*1.5)
#     upper_limit = q3 + (IQR*1.5)
#     df.loc[df[feature]<lower_limit,feature] = lower_limit
#     df.loc[df[feature]>upper_limit,feature] = upper_limit


# numerical_features_with_null = [feature for feature in numerical_features if df[feature].isnull().sum()]

# for feature in numerical_features_with_null:
#     mean_value = df[feature].mean()
#     df[feature].fillna(mean_value,inplace=True)

# df['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)


# df['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)
# df['RainTomorrow'].value_counts().plot(kind='bar')
# def encode_data(feature_name):
#     mapping_dict = {}
#     unique_values = list(df[feature_name].unique())
#     for idx in range(len(unique_values)):
#         mapping_dict[unique_values[idx]] = idx
#     print(mapping_dict)
#     return mapping_dict

# df['WindGustDir'].replace(encode_data('WindGustDir'),inplace = True)
# df['WindDir9am'].replace(encode_data('WindDir9am'),inplace = True)
# df['WindDir3pm'].replace(encode_data('WindDir3pm'),inplace = True)
# df['Location'].replace(encode_data('Location'), inplace = True)


# target = 'RainTomorrow'
# df = df[['Location','month', 'MaxTemp','RainTomorrow', 'Rainfall', 'WindGustSpeed', 'Humidity9am', 'Pressure3pm']]
# y = df[target]
# X = df.drop([target],axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# # smo = SMOTE()
# # X_sm, y_sm = smo.fit_resample(X_train, y_train)
# # print('Classes échantillon SMOTE :', dict(pd.Series(y_sm).value_counts()))

# classifier = GradientBoostingClassifier()
# # classifier.fit(X_sm, y_sm)
# classifier.fit(X_test, y_test)
# with open('classifier.pickle', 'wb') as f:
#     pickle.dump(classifier, f)
# df = pd.read_csv('./weatherAUS.csv')


# df.Location.unique()
# df['Date'] = pd.to_datetime(df['Date'])
# df['Year'] = df['Date'].apply(lambda x: x.year)
# data = df


# df['year'] = df['Date'].dt.year
# df['month'] = df['Date'].dt.month
# df['day'] = df['Date'].dt.day


# d = data[data['Location'] == region]
# d = d[['Date', 'Rainfall']]
# d.dropna(inplace=True)

# d.columns = ['ds', 'y'] 

# mod = NeuralProphet()
# model = mod.fit(d, freq='D')
# with open('ts.pickle', 'wb') as f:
#     pickle.dump(mod, f)



prd = 0
option =  st.selectbox(
    'Select how many days/weeks you want the forecast on',
 ('Tomorrow', 'Next week', 'Next Month'))

if option == 'Tomorrow':
    prd = 1
elif option == 'Next Month':
    prd = 30
elif option == 'In 3 Month':
    prd = 90
df_test = pd.DataFrame(columns=['Location', 'MaxTemp','RainTomorrow', 'Rainfall', 'WindGustSpeed', 'Humidity9am', 'Pressure3pm'])
if prd == 1:
  option =  st.write(
    'how is the weather today in Australia?'
  )
  
  loc_dict = {'Albury': 0, 'BadgerysCreek': 1, 'Cobar': 2, 'CoffsHarbour': 3, 'Moree': 4, 'Newcastle': 5, 'NorahHead': 6, 'NorfolkIsland': 7, 'Penrith': 8, 'Richmond': 9, 'Sydney': 10, 'SydneyAirport': 11, 'WaggaWagga': 12, 'Williamtown': 13, 'Wollongong': 14, 'Canberra': 15, 'Tuggeranong': 16, 'MountGinini': 17, 'Ballarat': 18, 'Bendigo': 19, 'Sale': 20, 'MelbourneAirport': 21, 'Melbourne': 22, 'Mildura': 23, 'Nhil': 24, 'Portland': 25, 'Watsonia': 26, 'Dartmoor': 27, 'Brisbane': 28, 'Cairns': 29, 'GoldCoast': 30, 'Townsville': 31, 'Adelaide': 32, 'MountGambier': 33, 'Nuriootpa': 34, 'Woomera': 35, 'Albany': 36, 'Witchcliffe': 37, 'PearceRAAF': 38, 'PerthAirport': 39, 'Perth': 40, 'SalmonGums': 41, 'Walpole': 42, 'Hobart': 43, 'Launceston': 44, 'AliceSprings': 45, 'Darwin': 46, 'Katherine': 47, 'Uluru': 48}
  df_test['Location'] = float(loc_dict[region])
  df_test['MaxTemp'] = st.slider('Temperature', -10, 50,5)
  df_test['Humidity9am'] = st.slider('Humidity', 0, 100, 10)
  df_test['Pressure3pm'] = st.slider('Pressure', -10, 50,5)
  df_test['WindGustSpeed'] = st.slider('Wind', 5, 150, 10)
  df_test['Rainfall'] = st.slider('Rainfall', 0, 500,20)
  df_test['month'] = 8
  df_test
else:
  target = st.selectbox('predict',
    ('MaxTemp', 'Rainfall'))
  month = 9 if prd == 30 else 11

if st.button('Forecast'):
  if prd == 1:
    with open('classifier.pickle') as f:
      classifier = pickle.load(f)
    y_pred = classifier.predict(X_test)
    y_pred[0]
    # st.write('Raining tomorrow' if y_pred['RainTomorrow'] == 1 else 'Not raining tomorrow')
  else:
    with open('ts.pickle') as f:
      mod = pickle.load(f)
    future = mod.make_future_dataframe(d, periods=prd, n_historic_predictions=True)
    prediction = mod.predict(future)
    forecast = mod.plot(prediction)
    st.write('Predicted ' + target + ' in ' + str(prd) + 'days is:' + str(prediction))



    
    