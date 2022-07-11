import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import time
import math
from sklearn.model_selection import train_test_split
from neuralprophet import NeuralProphet
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def preproc(df, type, useGeo):
    if useGeo:
        print(df.shape)
        df_coord = pd.read_csv('./data/coord.csv')
        df = pd.merge(df, df_coord,on=['Location'], how='right')
        print(df.shape)

    
    if type == 'timeseries':
        df.Location.unique()
        df.columns
        df['Date'] = pd.to_datetime(df['Date'])
        plt.plot(df['Date'], df['Temp3pm'])
        df['Year'] = df['Date'].apply(lambda x: x.year)
        data = df
        # df = df[df['Year']<=2015]
        # plt.plot(df['Date'], df['Temp3pm'])
        # plt.title('Rainfall')
        # plt.show()
        # plt.plot(df['Date'], df['Rainfall'])
        # plt.title('Temperature')
        # plt.show()
        
    else:  ## features
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'].dtype    
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df.drop('Date', axis = 1, inplace = True)


        categorical_features = [column_name for column_name in df.columns if df[column_name].dtype == 'O']
        print("Number of Categorical Features: {}".format(len(categorical_features)))
        print("Categorical Features: ",categorical_features)

        numerical_features = [column_name for column_name in df.columns if df[column_name].dtype != 'O']
        print("Number of Numerical Features: {}".format(len(numerical_features)))
        print("Numerical Features: ",numerical_features)
        # `Checking for Null values:`
        print(df[categorical_features].isnull().sum())
        # list of categorical features which has null values:

        categorical_features_with_null = [feature for feature in categorical_features if df[feature].isnull().sum()]
        # Filling the missing(Null) categorical features with most frequent value(mode)`

        for each_feature in categorical_features_with_null:
            mode_val = df[each_feature].mode()[0]
            df[each_feature].fillna(mode_val,inplace=True)
        df[categorical_features].isnull().sum()
        # checking null values in numerical features

        df[numerical_features].isnull().sum()
        # plt.figure(figsize=(15,10))
        # sns.heatmap(df[numerical_features].isnull(),linecolor='white')
        # visualizing the Null values in Numerical Features:

        # df[numerical_features].isnull().sum().sort_values(ascending = False).plot(kind = 'bar')
        # checking for outliers using Box Plot:
        # sns.pairplot(df, hue='RainTomorrow')
        # for feature in numerical_features:
        #     plt.subplot(21,21,i)
        #     sns.boxplot(df[feature])
        #     plt.title(feature)

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
        # for feature in numerical_features:
            # plt.figure(figsize=(10,10))
            # sns.boxplot(df[feature])
            # plt.title(feature)

        # list of numerical Features with Null values:

        numerical_features_with_null = [feature for feature in numerical_features if df[feature].isnull().sum()]
        # Filling null values uisng mean: 

        for feature in numerical_features_with_null:
            mean_value = df[feature].mean()
            df[feature].fillna(mean_value,inplace=True)
        df.isnull().sum()
        df.head()
        sns.pairplot(df, hue='RainToday')
        plt.show()
        df['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)


        df['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)
        df['RainTomorrow'].value_counts().plot(kind='bar')
        def encode_data(feature_name):
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

        
        data = df
        dataout = 'data_'+ type  +  '.pkl' if not useGeo else 'score_'+ type + '_geo.pkl'
        with open(dataout, "wb") as f:
            pickle.dump(data, f)
    return data

def train(m, data, target, useGeo):
    start_time = time.time()
    
    if m == 'fbprophet':
        
        if target == 'RainTomorrow':
            target = 'Rainfall'
        # data = data.drop_duplicates(subset=['ds'], keep='first')
        for r in ['Albany', 'Brisbane']: #data.Location.unique(): 
            d = data[data['Location'] == r]## if r else df
            d = d[d['Year']<=2013]
            d = d[['Date', target]]
            d.dropna(inplace=True)
            # data = data.drop_duplicates()
            
            d.columns = ['ds', 'y'] 
            print(d.shape)
            dataout = 'data_'+ 'timeseries_2012'  + r+  '.pkl'
            with open(dataout, "wb") as f:
                pickle.dump(d, f)
            with open(dataout, 'rb') as f:
                d = pickle.load(f)
            modelout = 'model_'+ m + '_2012_' + target + '_'+ r + '.pkl'
            
            # mod = NeuralProphet()
            # model = mod.fit(d, freq='D')#, epochs=1000)
            
            # with open(modelout, "wb") as f:
            #     pickle.dump(mod, f)
            with open(modelout, 'rb') as f:
                mod = pickle.load(f)
            future = mod.make_future_dataframe(d, periods=6*365)
            
            forecast = mod.predict(future)
             # with open(dataout, "wb") as f:
            #     pickle.dump(d, f)
            #print('forecast')
            #print(forecast.head())
            dfcompare = data[(data['Location'] == r)&(data['Year']>2012)]
            dfcompare = dfcompare[['Date','Year', 'RainToday', 'RainTomorrow','Rainfall', 'Temp3pm']]
            forecast['Rainfall_pred'] = forecast['yhat1']
            forecast['RainToday_pred'] = forecast['yhat1'].apply(lambda x: 1 if x >1 else 0)
                # rtomorrow_tgt = data['RainTomorrow'][0:forecast.shape[0]]
                # rtoday_tgt = data['RainToday'][0:forecast.shape[0]]
                # #rtomorrow_tgt = np.append(rtomorrow_tgt, 0)
                # forecast['RainTomorrow'] = rtomorrow_tgt
                # forecast['RainToday'] = rtoday_tgt

            def confmat(df: pd.DataFrame, col1: str, col2: str):
    
                return (
                        df
                        .groupby([col1, col2])
                        .size()
                        .unstack(fill_value=0)
                        )  
            print(forecast)
            
            forecast = forecast.rename(columns={'ds': 'Date'})
            print(dfcompare.columns)
            print(forecast.columns)
            dfcompare['rain2d'] = dfcompare.apply(lambda r: 1 if (r['RainTomorrow']==1 or r['RainToday']==1) else 0, axis=1)

            dfcompare = pd.merge(dfcompare, forecast, on=['Date'], how='left')
            cfm = confmat(dfcompare, 'RainToday', 'RainToday_pred')
            print(cfm)
            print(dfcompare.columns)
            esterr =dfcompare.apply(lambda r: (r['Rainfall'] - r['Rainfall_pred'])/r['Rainfall'] if r['Rainfall']> 0 else 0, axis=1)
            print(esterr)
            #print(forecast['yhat1'].tolist())
            # plot1 = mod.plot(forecast)
            # plt2 = mod.plot_components(forecast)
            scoreout = 'score_'+ m + '_' + target + '_'+ r + '.pkl'
            with open(scoreout, 'wb') as file:
                res = {
                    'model_type': 'timeseries',
                    'model': m,
                    'target': target,
                    'est_err': esterr,
                    'useGeo': 'na', 
                    'processing_time': 'few sec',
                    'accuracy_score': (cfm.iloc[0,0] + cfm.iloc[1,1])/forecast.shape[0],
                    'classifier_score_train': 'na',
                    'classifier_score_test': 'na'
                }
                pickle.dump(res, file)
            print(res)
    elif m == 'logreg':
        # `Spliting data into input features and label`
        if target == 'Temp3pm':
            def tmpClass(tmp):
                if tmp <= 18:
                    return 0
                elif tmp >10 and tmp <=20:
                    return 1
                elif tmp > 20 and tmp <=30:
                    return 2
                elif tmp > 30:
                    return 3
            data['TempClass'] = data['Temp3pm'].apply(tmpClass)
            target = 'TempClass'
        y = data[target]
        X = data.drop([target],axis=1)
        print('logreg')
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # with open('scaler.pkl', 'wb') as file:
        #     pickle.dump(scaler, file)
        
        classifier = LogisticRegression(solver='liblinear', random_state=0)
        classifier.fit(X_train, y_train)
        end_time = time.time()
        modelout = 'score_'+ m + '.pkl' if not useGeo else 'score_'+ m + '_geo.pkl'
        with open(modelout, 'wb') as file:
            pickle.dump(classifier, file)
        y_pred = classifier.predict(X_test)
        acc_score = accuracy_score(y_test,y_pred)
        classifier_score_train = classifier.score(X_train, y_train)
        classifier_score_test = classifier.score(X_test, y_test)
        t = 'features'
        r='all'
        
        scoreout = 'score_'+ m + '.pkl' if not useGeo else 'score_'+ m + '_geo.pkl'
        with open(scoreout, 'wb') as file:
            res = {
                'model_type': t,
                'model': m,
                'target': target,
                'useGeo': useGeo, 
                'processing_time': end_time - start_time,
                'accuracy_score': acc_score,
                'classifier_score_train': classifier_score_train,
                'classifier_score_test': classifier_score_test
            }
            pickle.dump(res, file)
            print(res)
    elif m == 'randomforest':  
        print('random forest...')
        if target == 'Temp3pm':
            def tmpClass(tmp):
                if tmp <= 18:
                    return 0
                elif tmp >10 and tmp <=20:
                    return 1
                elif tmp > 20 and tmp <=30:
                    return 2
                elif tmp > 30:
                    return 3
            data['TempClass'] = data['Temp3pm'].apply(tmpClass)
            target = 'TempClass'
        y = data[target]
        X = data.drop([target],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
        start_time = time.time()
        classifier=RandomForestClassifier()
        classifier.fit(X_train,y_train)
        end_time = time.time()
        
        modelout = 'score_'+ m + '.pkl' if not useGeo else 'score_'+ m + '_geo.pkl'
        with open(modelout, 'wb') as file:
            pickle.dump(classifier, file)
        y_pred = classifier.predict(X_test)
        acc_score = accuracy_score(y_test,y_pred)
        classifier_score_train = classifier.score(X_train, y_train)
        classifier_score_test = classifier.score(X_test, y_test)
        t = 'features'
        r='all'
        scoreout = 'score_'+ m + '.pkl' if not useGeo else 'score_'+ m + '_geo.pkl'
        with open(scoreout, 'wb') as file:
            res = {
                'model_type': t,
                'model': m,
                'target': target,
                'useGeo': useGeo, 
                'processing_time': end_time - start_time,
                'accuracy_score': acc_score,
                'classifier_score_train': classifier_score_train,
                'classifier_score_test': classifier_score_test
            }
            pickle.dump(res, file)
            print(res)
        # sns.relplot(x=df['long'], y=df['lat'], hue=df['lat_area'], size=df['MaxTemp'])#abs(dfwithCoord['MaxTemp'] - dfwithCoord['MinTemp']))

    end_time = time.time()
    print("Train time {}: {}".format(m, end_time - start_time))
  

def explore(t):
    print('Data exploration ' + t)
    if t == 'timeseries':
        with open('data_timeseries.pkl', 'rb') as f:
            data = pickle.load(f)
    elif t == 'features':
        with open('data_features.pkl', 'rb') as f:
            data = pickle.load(f)   
    plt.subplot(3,1)
    plt.plot(data['Date'], data['Rainfall'])
    plt.title('Rainfall')
    plt.plot(data['Date'], data['Temp3pm'])
    plt.title('Temp3pm')
    plt.plot(data['Date'], data['Humidity3pm'])

def perf(t, m, ):
    res = {}
    if t == 'timeseries':
        print('perf timeseries')
        if m == 'fbprophet':
            print(m)
    elif t == 'features':
        print('perf features')
        if m == 'logreg':
            print(m)
    res = m
    return res
def pred(t, m):
    res = {}
    print('PRED {}', t)
    if t == 'timeseries':
        if m == 'fbprophet':
            print(m)
    elif t == 'features':
        if m == 'logreg':
            print(m)
    res = m
    return res