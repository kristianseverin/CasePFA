# CasePFA
This repo contains Kristian Severin's case for a job interview at PFA. 

The case features a dataset of crimes commmmited in San Francisco 'sf_data.csv' and the districts in which the crimes were commited 'sf_districts.csv'.
After merging the dataframes by id's it became clear that only a proportion of the id's had district information. For various reasons it was chosen to only perform analysis on the id's with full data available. 

Timeseries XGBoost regression models were created and tested. Target variable was created and made numeric by counting crime occurences per day. Different models were tested:

XGBoostRegressor: using number of crimes per day as target and timeseries components as numerical inputs. 
XGBoostCatAndNum: using number of crimes per day as target and timeseries components as numerical inputs. Categorical variables were made compatible to use as inputs by making an embedding model. 

Models can be run from the command line using the runModels.py script as such:
python runModels.py -d path/to/csv/with/numdata -m XGBoostRegressor -s /path/to/folder/where/plot/should/be/saved/

-d: path to the data found in src/Data/. Should point to dfCatAndNum.csv if the model with both numerical and categorical variables should be run. Should point to dfNumeric.csv if the numerical regressor is to be run.

-m name of the XGBoost model that is to be run. Can be either XGBoostRegressor or XGBoostCatAndNumRegressor

-s path to save a plot showing the forecasted values adjacent to the 5 months leading up to the forecast. 


I have kept the 'XGBoost.ipynb' to let potentially interested people get an insight into my working process. Here you can find different exploratory plots + different models tested with no hyperparamter tuning. I think this is a good resource to understand my step by step process :) 
