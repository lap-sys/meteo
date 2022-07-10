import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import time
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
        # plt.plot(df['Date'], df['Temp3pm'])
        # plt.show()
        df['Year'] = df['Date'].apply(lambda x: x.year)
        data = df
        # df = df[df['Year']<=2015]
        # plt.plot(df['Date'], df['Temp3pm'])
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

        df[numerical_features].isnull().sum().sort_values(ascending = False).plot(kind = 'bar')
        #checking for outliers using Box Plot:

        # for feature in numerical_features:
        #     plt.figure(figsize=(10,10))
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
        numerical_features_with_null
        # Filling null values uisng mean: 

        for feature in numerical_features_with_null:
            mean_value = df[feature].mean()
            df[feature].fillna(mean_value,inplace=True)
        df.isnull().sum()
        df.head()

        df['RainToday'].replace({'No':0, 'Yes': 1}, inplace = True)

        # pd.get_dummies(df['RainToday'],drop_first = True)

        df['RainTomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)
        # print(df['RainTomorrow'].value_counts())#.plot(kind='bar')
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
        for r in data.Location.unique(): 
            # d = data[data['Location'] == r]## if r else df
            # d = d[['Date', target]]
            # d.dropna(inplace=True)
            # # data = data.drop_duplicates()
            # d.columns = ['ds', 'y'] 
            # print(d.shape)
            dataout = 'data_'+ 'timeseries_'  + r+  '.pkl'
            # with open(dataout, "wb") as f:
            #     pickle.dump(d, f)
            with open(dataout, 'rb') as f:
                d = pickle.load(f)
            modelout = 'model_'+ m + '_' + target + '_'+ r + '.pkl'
            
            # mod = NeuralProphet()
            # model = mod.fit(d, freq='D')#, epochs=1000)
            
            scoreout = 'score_'+ m + '_' + target + '_'+ r + '.pkl'
            # with open(modelout, "wb") as f:
            #     pickle.dump(mod, f)
            with open(modelout, 'rb') as f:
                mod = pickle.load(f)
            future = mod.make_future_dataframe(d, periods=6*365)
            print(future)
            forecast = mod.predict(future)
            print('forecast')
            print(forecast.head())
            plot1 = mod.plot(forecast)
            plt2 = mod.plot_components(forecast)

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