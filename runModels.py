from Models import XGBoostRegressor
import pandas as pd
import argparse

# load data
#df = pd.read_csv('/Users/kristian/Documents/CasePFA/Models/Data/dfNumeric.csv')

# load data using arguments
#df = pd.read_csv(args.df)

# set datetime as index
#df['datetime'] = pd.to_datetime(df['datetime'])

# split the data
#train = df[df['datetime'] < df['datetime'].quantile(0.8)]
#test = df[df['datetime'] >= df['datetime'].quantile(0.8)]

# set datetime as index
#train = train.set_index('datetime')
#test = test.set_index('datetime')

#FEATURES = ['year', 'month', 'day', 'hour', 'minute', 'weekday']
#TARGET = 'num_crimes'

##X_train = train[FEATURES]
#y_train = train[TARGET]

#X_test = test[FEATURES]
#y_test = test[TARGET]

# create an instance of the XGBoostRegressor class
#xgboost = XGBoostRegressor(df, X_train, y_train, X_test, y_test)

# call the xgboost_regressor method
#xgboost.xgboost_regressor()

#print the forecast dataframe with the predicted number of crimes
#print(xgboost.forecast(xgboost.xgboost_regressor()))

# plot the forecast
#xgboost.plot_forecast(xgboost.forecast(xgboost.xgboost_regressor()), xgboost.xgboost_regressor())

# plot the actual vs. predicted using df as the actual values
#xgboost.plot_actual_vs_predicted(xgboost.xgboost_regressor().predict(X_test), xgboost.xgboost_regressor(), df)


# make arguments for the forecast method



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the XGBoostRegressor model')
    parser.add_argument('-d','--df', type=str, help='The dataframe to be used in the model')
    parser.add_argument('-m', '--model', type=str, help= 'The model to be used 1. XGBoostRegressor , default = XGBoostRegressor')
    parser.add_argument('-s', '--save', type=str, help='The path to save the forecast plot') 

    args = parser.parse_args()

    # load data using arguments
    df = pd.read_csv(args.df)

    # set datetime as index
    df['datetime'] = pd.to_datetime(df['datetime'])

    # split the data
    train = df[df['datetime'] < df['datetime'].quantile(0.8)]
    test = df[df['datetime'] >= df['datetime'].quantile(0.8)]

    # set datetime as index
    train = train.set_index('datetime')
    test = test.set_index('datetime')

    FEATURES = ['year', 'month', 'day', 'hour', 'minute', 'weekday']
    TARGET = 'num_crimes'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

   
    # create an instance of the XGBoostRegressor class
    xgboost = XGBoostRegressor(X_train, y_train, X_test, y_test)


    if args.model == 'XGBoostRegressor':
        # call the xgboost_regressor method
        #xgboost.xgboost_regressor()

        # plot the forecast using savepath = args.save
        xgboost.plot_forecast(xgboost.forecast(xgboost.xgboost_regressor()), args.save)        
    else:
        print('Invalid model')

        

        

        



# a valid command to run the script would be:
# python runModels.py -d /Users/kristian/Documents/CasePFA/Data/dfNumeric.csv -m XGBoostRegressor -s /Users/kristian/Documents/CasePFA/Results/    

