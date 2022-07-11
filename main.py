
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
            ['fbprophet'],
        'features':
           ['logreg', 'randomforest']
    }
    useGeos = [True, False]
    terms=['day','week']
    targets = ['RainTomorrow']#, 'Temp3pm']
    #######

    data = {}
    plots = {}
    for useGeo in useGeos:
        for t in modeltypes:
            data[t] = utils.preproc(df, t, useGeo)
            for m in models[t]:
                for tgt in targets:
                    utils.train(m, data[t], tgt, useGeo)

        
    


    
    