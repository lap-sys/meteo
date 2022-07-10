
import utils
import pandas as pd
import sys

predictionMode = len(sys.argv) > 1

if predictionMode:
    date = sys.argv[1]
    region = sys.argv[2]
    print('Méteo dans ' + date + ' jour(s) à '+ region)
    preds = utils.pred(date, region)
    print(preds)
else:
    df = pd.read_csv('./data/weatherAUS.csv')
    modeltypes = ['timeseries', 'features']
    models = {
        'timeseries': 
            ['fbprophet'],# 'sarimax'],
        'features': 
            ['logreg', 'randomforest']#, 'decision_tree', 'cat_boost']
    }
    useGeos = [True, False]
    # regions = [0, 1, 2, 3, 4] if useGeo else None
    terms=['day','week']
    targets = ['RainTomorrow', 'Temp3pm']
    dt = 'day'
    #######

    data = {}
    plots = {}
    for useGeo in useGeos:
        for t in modeltypes:
            data[t] = utils.preproc(df, t, useGeo)
            plots[t] = utils.explore(t)
            for m in models[t]:
                for tgt in targets:
                    utils.train(m, data[t], tgt, useGeo)

            
       


      
     