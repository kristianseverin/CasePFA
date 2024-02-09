import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

class XGBoostCatAndNumRegressor:
    def __init__(self, df, X_train, y_train, X_test, y_test):
        self.df = df
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def xgboost_regressor_cat_num(self):

        xg_reg_cat_num = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                  max_depth=5, alpha=10, n_estimators=10)

        xg_reg_cat_num .fit(self.X_train, self.y_train)

        preds = xg_reg_cat_num .predict(self.X_test)

        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        print("RMSE: %f" % (rmse))

        # Hyperparameter tuning
        param_grid = {
            'learning_rate': [0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5],
            'n_estimators': [10, 20, 30]
        }

        xg_reg_cat_num  = xgb.XGBRegressor()

        grid_search = GridSearchCV(estimator=xg_reg_cat_num, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

        grid_search.fit(self.X_train, self.y_train)

        print(grid_search.best_params_)

        xg_reg_cat_num = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                  max_depth=5, alpha=10, n_estimators=10)

        xg_reg_cat_num.fit(self.X_train, self.y_train)

        preds = xg_reg_cat_num.predict(self.X_test)

        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        print("RMSE: %f" % (rmse))
        return xg_reg_cat_num

    def forecast(self, xg_reg_cat_num):
        y_pred = xg_reg_cat_num.predict(self.X_test)

        dates = pd.date_range(start='2018-05-16', end='2018-10-31', freq='D')
        df_forecast = pd.DataFrame(dates, columns=['datetime'])


        df_forecast['category'] = self.df['category']
        df_forecast['resolution'] = self.df['resolution']
        df_forecast['label'] = self.df['label']
        df_forecast['district'] = self.df['district']



        df_forecast['year'] = df_forecast['datetime'].dt.year
        df_forecast['month'] = df_forecast['datetime'].dt.month
        df_forecast['day'] = df_forecast['datetime'].dt.day
        df_forecast['hour'] = df_forecast['datetime'].dt.hour
        df_forecast['minute'] = df_forecast['datetime'].dt.minute
        df_forecast['weekday'] = df_forecast['datetime'].dt.weekday
        df_forecast['category'] = df_forecast['category']
        df_forecast['resolution'] = df_forecast['resolution']
        df_forecast['label'] = df_forecast['label']
        df_forecast['district'] = df_forecast['district']

        y_forecast = xg_reg_cat_num.predict(df_forecast[['year', 'month', 'day', 'hour', 'minute', 'weekday', 'category', 'resolution', 'label', 'district']])

        df_forecast['num_crimes'] = y_forecast
        return df_forecast 

    def plot_forecast(self, df_forecast, savepath=None):

        fig, ax = plt.subplots(figsize=(15, 8))
        # the five months before '2018-03-16':'2018-05-15'
        self.y_test['2017-12-16':'2018-05-15'].plot(ax=ax, title='Number of crimes per day for the five months before the forecast')

        # the forecast
        df_forecast.set_index('datetime')['num_crimes'].plot(ax=ax, title='Number of crimes per day for the five months before the forecast')

        # legend
        ax.legend(['Observed', 'Forecast'])

        #plt.show() # show the plot in the console window

        # save the plot in Results folder using the save argument
        plt.savefig(savepath + 'forecastCatAndNum.png')

     



        
        
