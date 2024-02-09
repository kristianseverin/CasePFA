from Models import XGBoostRegressor, XGBoostCatAndNumRegressor, EmbeddingModel
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
    
    elif args.model == 'XGBoostCatAndNumRegressor':

        df_cat_num = pd.read_csv('/Users/kristian/Documents/CasePFA/Data/dfCatAndNum.csv')

        categorical_features = ['category', 'resolution', 'label', 'district']
        numerical_features = ['year', 'month', 'day', 'hour', 'minute', 'weekday']

        # define the sizes of embeddings for each categorical feature
        embd_sizes = [(categories, size) for categories, size in zip(df_cat_num[categorical_features].nunique(), [1]*len(categorical_features))]

        # instantiate the model
        emb_model = EmbeddingModel(embd_sizes, len(numerical_features))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(emb_model.parameters(), lr=0.01)

        # convert to tensor
        categorical_inputs = torch.tensor(df_cat_num[categorical_features].values, dtype = torch.long)
        numerical_inputs = torch.tensor(df_cat_num[numerical_features].values, dtype = torch.float32)
        targets = torch.tensor(df_cat_num['num_crimes'].values, dtype = torch.float32)
        targets = targets.view(-1, 1) # reshape the tensor


        # train model
        for epoch in range(100):
            optimizer.zero_grad()
            output = emb_model(categorical_inputs, numerical_inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            print(f'this is the loss: {loss}')
        
        # Extract embeddings from the trained model
        embeddings = [embedding.weight.detach().numpy() for embedding in emb_model.embd_layers]


        # pad all embeddings to get the size (36, 10)
        max_length = max([embedding.shape[0] for embedding in embeddings])
        for i, embedding in enumerate(embeddings):
            pad_length = max_length - embedding.shape[0]
            embeddings[i] = np.pad(embedding, ((0, pad_length), (0, 0)), 'constant', constant_values=0)


        # concatenate the embeddings
        X = np.concatenate(embeddings, axis=1)

        # get weight embeddings
    for col in categorical_features:
        weight_embeddings = col + '_embedding'
        weight_df = pd.DataFrame(emb_model.embd_layers[i].weight.detach().numpy(), columns = [col + '_embedding' + str(i) for i in range(1)])


    # create a dictionary that maps each embedding value to the original categorical level
    embedding_dict = {}
    for i, col in enumerate(categorical_features):
        weight_embeddings = col + '_embedding'
        weight_df = pd.DataFrame(emb_model.embd_layers[i].weight.detach().numpy(), columns = [col + '_embedding' + str(i) for i in range(1)])
        embedding_dict[col] = weight_df.idxmax(axis=1).to_dict()



    # make a dataframe with the embeddings
    df_embed = pd.DataFrame(X, columns = [col + '_embedding' + str(i) for i in range(1) for col in categorical_features])

    # assign the embeddings to the original categorical levels
    for col in categorical_features:
        for i in col:
            # the first embedding should be assigned to the first unique value of the categorical feature
            df_embed[col + '_embedding' + str(i)] = df_cat_num[col].map(embedding_dict[col])

    def map_embeddings_to_categories(df_embed, df_cat_num):
        # category
        cat_embed = df_embed['category_embedding0']
        unique_cat = df_cat_num['category'].unique()
        embedding_map_cat = {category: embedding for category, embedding in zip(unique_cat, cat_embed)}
        df_cat_num['category'] = df_cat_num['category'].map(embedding_map_cat)

        # resolution
        res_embed = df_embed['resolution_embedding0']
        unique_res = df_cat_num['resolution'].unique()
        embedding_map_res = {resolution: embedding for resolution, embedding in zip(unique_res, res_embed)}
        df_cat_num['resolution'] = df_cat_num['resolution'].map(embedding_map_res)

        # label
        lab_embed = df_embed['label_embedding0']
        unique_lab = df_cat_num['label'].unique()
        embedding_map_lab = {label: embedding for label, embedding in zip(unique_lab, lab_embed)}
        df_cat_num['label'] = df_cat_num['label'].map(embedding_map_lab)

        # district
        dist_embed = df_embed['district_embedding0']
        unique_dist = df_cat_num['district'].unique()
        embedding_map_dist = {district: embedding for district, embedding in zip(unique_dist, dist_embed)}
        df_cat_num['district'] = df_cat_num['district'].map(embedding_map_dist)

        return df_cat_num

    # use function to map embeddings to categories
    df_cat_num = map_embeddings_to_categories(df_embed, df_cat_num)

    # set datetime as index
    df_cat_num['datetime'] = pd.to_datetime(df_cat_num['datetime'])

    # make train and test by finding the 80th percentile
    train_cat_num = df_cat_num[df_cat_num['datetime'] < df_cat_num['datetime'].quantile(0.8)]
    test_cat_num = df_cat_num[df_cat_num['datetime'] >= df_cat_num['datetime'].quantile(0.8)]

    # set datetime as index
    train_cat_num = train_cat_num.set_index('datetime')
    test_cat_num = test_cat_num.set_index('datetime')

    # define features and target
    FEATURES_CAT_NUM = ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'category', 'resolution', 'label', 'district']  # order is important
    TARGET_CAT_NUM = 'num_crimes'

    # make x and y
    X_train_cat_num = train_cat_num[FEATURES_CAT_NUM]
    y_train_cat_num = train_cat_num[TARGET_CAT_NUM]

    X_test_cat_num = test_cat_num[FEATURES_CAT_NUM]
    y_test_cat_num = test_cat_num[TARGET_CAT_NUM]

    # create an instance of the XGBoostRegressor class for the categorical and numerical features
    xgboost_cat_num = XGBoostCatAndNumRegressor(df_cat_num, X_train_cat_num, y_train_cat_num, X_test_cat_num, y_test_cat_num)

    # plot the forecast using savepath = args.save
    xgboost_cat_num.plot_forecast(xgboost_cat_num.forecast(xgboost_cat_num.xgboost_regressor_cat_num()), args.save) 

else:
     print('The model is not available')
    


        






        

        

        



# a valid command to run the script would be:
# python runModels.py -d /Users/kristian/Documents/CasePFA/Data/dfNumeric.csv -m XGBoostCatAndNumRegressor -s /Users/kristian/Documents/CasePFA/Results/    

