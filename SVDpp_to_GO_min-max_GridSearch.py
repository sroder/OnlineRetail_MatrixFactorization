import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams["figure.figsize"] = [16,9]

from surprise import SVD,NMF,SVDpp,evaluate
from surprise.dataset import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection.search import GridSearchCV
from surprise.model_selection import train_test_split
from surprise.model_selection.search import GridSearchCV

dfs = pd.read_excel('OnlineRetail.xlsx', sheet_name='OnlineRetail')
#remove canceled orders
dfs = dfs[dfs['Quantity']>0]
#remove rows where customerID are NA
dfs.dropna(subset=['CustomerID'],how='all',inplace=True)

###### Multiplying Quantity and UnitPrice columns to get a new column : AmountSpend########
dfs['AmountSpend'] = dfs['Quantity']*dfs['UnitPrice']
matrix_setup = dfs[['StockCode','CustomerID','AmountSpend']]

# removing outliers
matrix_setup = matrix_setup[(matrix_setup['AmountSpend'] < 5000)]
print "Number of unique products bought: ",matrix_setup['StockCode'].nunique()
print "Number of unique customers:", matrix_setup['CustomerID'].nunique() 
#Mean
matrix_setup['Mean_amount'] = matrix_setup.groupby(['StockCode','CustomerID'])['AmountSpend'].transform(np.mean)
#remove duplicates
mtarix_toGO = matrix_setup.drop_duplicates(subset = ['StockCode','CustomerID'], keep = 'first')

# need to normalize the Mean_Amount column(which is going to be predicted) otherwise prediction takes a lot of time and also bad results
#mtarix_toGO.loc[mtarix_toGO['Mean_amount'] == 0 ] = 0.001
#mtarix_toGO['Log_Mean_amount'] = np.log(mtarix_toGO['Mean_amount'])

min_amt = min(mtarix_toGO['Mean_amount'])
max_amt = max(mtarix_toGO['Mean_amount'])
print 'Minimum mean amount',min_amt
print 'Maximum mean amount',max_amt

# Normalized data [0,1]
mtarix_toGO['Norm_Tot_Amnt']= (mtarix_toGO['Mean_amount'] -min_amt)/max_amt
#lower_bound = min(mtarix_toGO['Log_Mean_Amount'])
#upper_bound = max(mtarix_toGO['Log_Mean_Amount'])
#print lower_bound
#print upper_bound
# Remove the outliers
dfx=mtarix_toGO[mtarix_toGO['Norm_Tot_Amnt'] <= 0.4]
lower_bound = min(dfx['Norm_Tot_Amnt'])
upper_bound = max(dfx['Norm_Tot_Amnt'])
print 'Lower Bound normalized spending =',lower_bound
print 'Upper Bound normalized spending =',upper_bound
print 'Number of Transactions remaining after removing Outliers::',mtarix_toGO.shape[0]

#define the reader  with  upper and lower bounds , also now we are predicting Normalized Total Amount column
reader_x = Reader(rating_scale = (lower_bound,upper_bound))
data = Dataset.load_from_df(df=dfx[['CustomerID','StockCode','Norm_Tot_Amnt']],reader=reader_x)


#for i in range(9):
#    print (data.raw_ratings[0][2] - data.df['Log_Mean_amount'][0])

print 'difference in processed and pre-processed dataset = ',(data.raw_ratings[0][2] - data.df['Norm_Tot_Amnt'][0])

import time
start_time = time.time()


#param_grid = {'n_factors':[2,5,10,50],'n_epochs': [10,50,100], 'lr_bu': [0.1,0.01,0.001,0.0001],'lr_bi': [0.1,0.01,0.001,0.0001],'reg_bi': [0.1,0.01,0.001,0.0001],'reg_bu': [0.1,0.01,0.001,0.0001],'reg_qi': [0.1,0.01,0.001,0.0001],'reg_pu': [0.1,0.01,0.001,0.0001]}
param_grid = {'n_factors':[5,10,50,100],'n_epochs': [5,10,20,50,100], 'lr_all': [0.1,0.01,0.001],'reg_all': [0.1,0.01,0.001}
grid_search = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=1)

grid_search.fit(data)

print 'best RMSE score'
print(grid_search.best_score['rmse'])

print 'combination of parameters that gave the best RMSE score'
print(grid_search.best_params['rmse'])

print 'best MAE score'
print(grid_search.best_score['mae'])

print 'combination of parameters that gave the best MAE score'
print(grid_search.best_params['mae'])

print("--- %s seconds for GridSearch---" % (time.time() - start_time))


results_df = pd.DataFrame.from_dict(grid_search.cv_results)
print results_df


