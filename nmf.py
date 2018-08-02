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
print("Number of products bought: ",matrix_setup['StockCode'].nunique())
print("Number of customers:", matrix_setup['CustomerID'].nunique() )
#Mean
matrix_setup['Mean_amount'] = matrix_setup.groupby(['StockCode','CustomerID'])['AmountSpend'].transform(np.mean)
#remove duplicates
mtarix_toGO = matrix_setup.drop_duplicates(subset = ['StockCode','CustomerID'], keep = 'first')
# need to normalize the Mean_Amount column(which is going to be predicted) otherwise prediction takes a lot of time and also bad results
mtarix_toGO.loc[mtarix_toGO['Mean_amount'] == 0 ] = 0.001
#mtarix_toGO['Log_Mean_Amount'] = np.log(mtarix_toGO['Mean_Amount'])
mtarix_toGO['Log_Mean_amount'] = np.log(mtarix_toGO['Mean_amount'])
#min_amt = min(mtarix_toGO['Mean_amount'])
#max_amt = max(mtarix_toGO['Mean_amount'])
#print min_amt
#print max_amt
# Normalized data [0,1]
#mtarix_toGO['Norm_Tot_Amnt']= (mtarix_toGO['Mean_amount'] -min_amt)/max_amt
lower_bound = min(mtarix_toGO['Log_Mean_Amount'])
upper_bound = max(mtarix_toGO['Log_Mean_Amount'])
#print lower_bound
#print upper_bound
# Remove the outliers
#dfx=mtarix_toGO[mtarix_toGO['Norm_Tot_Amnt'] <= 0.4]
#lower_bound = min(dfx['Norm_Tot_Amnt'])
#upper_bound = max(dfx['Norm_Tot_Amnt'])
print ('Lower Bound',lower_bound)
print ('Upper Bound',upper_bound)

#define the reader  with  upper and lower bounds , also now we are predicting Normalized Total Amount column
reader_x = Reader(rating_scale = (lower_bound,upper_bound))
data = Dataset.load_from_df(df=dfx[['CustomerID','StockCode','Norm_Tot_Amnt']],reader=reader_x)


for i in range(9):
    print (data.raw_ratings[i][2] - data.df['Log_Mean_amount'][i])



algo = SVD(lr_all=0.01, reg_all= 0.1, n_factors=50, n_epochs=10)

trainset, testset = train_test_split(data, test_size=.25, random_state=27, shuffle=True)
import time
start_time = time.time()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
# Predictions
predictions = algo.test(testset)
print("--- %s seconds ---" % (time.time() - start_time))

# Then compute RMSE, earlier RMSE: 0.4752(without subtraction of constant from data.raw_ratings)

from surprise import accuracy
accuracy.rmse(predictions)
accuracy.mae(predictions)

test_list = []
for i in testset:
    test_list.append(i[-1])

prediction_list = []
for i in predictions:
    prediction_list.append(i[3])
	
plt.scatter(prediction_list,test_list)
#plt.xlim(0,0.20)
#plt.ylim(0,0.20)
plt.show()

#param_grid = {'n_factors':[2,5,10,50],'n_epochs': [10,50,100,500], 'lr_bu': [0.1,0.01,0.001,0.0001],'lr_bi': [0.1,0.01,0.001,0.0001],'reg_bi': [0.1,0.01,0.001,0.0001],'reg_bu': [0.1,0.01,0.001,0.0001],'reg_qi': [0.1,0.01,0.001,0.0001],'reg_pu': [0.1,0.01,0.001,0.0001]}
#grid_search = GridSearchCV(NMF, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=1)

#grid_search.fit(data)
# best RMSE score
#print(grid_search.best_score['rmse'])

# combination of parameters that gave the best RMSE score
#print(grid_search.best_params['rmse'])

# best MAE score
#print(grid_search.best_score['mae'])

# combination of parameters that gave the best MAE score
#print(grid_search.best_params['mae'])

#print("--- %s seconds ---" % (time.time() - start_time))


#results_df = pd.DataFrame.from_dict(grid_search.cv_results)
#results_df

