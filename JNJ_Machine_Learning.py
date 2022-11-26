import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import OneHotEncoder
import copy
import collections
from statsmodels.tsa.seasonal import seasonal_decompose
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

###Look into making data cyclical as in months, seasons, weeks at the beginning and end of year should be related
#Look into autocorrelation, ie. time delay
#Look into differences between date values when windowing, ie. weekends and holidays
### Include high, low, open data
#More feature engineering like considering if a holiday just happened
###For feature engineering look into using technical analysis indicators
###Should I implement dynamic learning (ie. learn 1 data point ahead, retrain model using this new data point, repeat), will this fix the large values
#Use AUTOML/H20 for finding a ML Model/Neural Network
#Include a comparison from March 2020 and onward to check for quantitative easing
###Use one hot encoder for weekday,month, season etc.
#Look into multicolinearity
###Code RSI for different training set sizes
###Include time differences
###Take out year
###Take out time float variable
#Fit data to only set where price is not outside training data
#Include actual timestamp in model fitting
#loop through look backsizes to find best fit and in those loops find the best parameters




#SMA - Simple Moving Averge
#EMA - Exponential Moving Average
#MACD - Moving Average Convergence Divergence
#RSI - Relative Strength Index

#container for the date data from the jnj stock price csv
jnj_date = []

#container for the price data from the jnj stock price csv
jnj_prices = []

#opening the csv file and then skipping the first row as it is a header
with open('JNJ.csv', newline = '') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter = ',')
	next(csv_reader)

#looping through each row of the data
	for row in csv_reader:
#split the data data into a float timestamp so that it is easier to work with
		date = datetime.strptime(row[0], '%Y-%m-%d')
		float_time = date.timestamp()
#get the year and month for each date
		year = date.year
		month = date.month

#define the seasons as quarters and define them numerically
		if 1 <= month <= 3:
			season = 1
		elif 4 <= month <= 6:
			season = 2
		elif 7 <= month <= 9:
			season = 3
		elif 10 <= month <= 12:
			season = 4	
#get the week and weekday for each date
		week = date.isocalendar()[1]
		weekday = date.weekday()

#appending all of the date features to the date data list
		jnj_date.append([float_time,year,season,month,week,weekday])

#all of the price and volume data from the csv file
		volume = float(row[6])
		
		open_price = float(row[1])

		high_price = float(row[2])

		low_price = float(row[3])

		close_price = float(row[4])

#appending all of the price features to the price data list
		jnj_prices.append([volume, open_price, high_price, low_price, close_price])




#size of the moving average technical indicators, ie. SMA and EMA
MA_size = 20

#SMA

def SMA_func(period, stock):
#Holds the summed value of all the close price values in the SMA period
	SMA_sum = 0
#Holds all of the SMA values calculated from the close price values
	SMA_list = []
#Starting at the end of a single period length so that each data point can be used to calculate the SMA
	for i in range(period,len(stock)):
		for j in range(period):
#Summing up each close price value within the period
			SMA_sum += stock[i-j][-1]
#Calculating the SMA and appending it to the list of SMA values and then resetting the summed value holder back to zero for the next iteration
		SMA_list.append(SMA_sum/period)
		SMA_sum = 0
	return np.array(SMA_list)

SMA = SMA_func(MA_size, jnj_prices)



#EMA

#Smoothing factor used in the EMA
smoothing_factor = 2

def EMA_func(smoothing, period, stock):

	weight = smoothing/(period+1)

#Calculating the first value of the EMA using the first SMA value
	EMA_list = [stock[period][-1]*weight + SMA_func(period,stock)[0]*(1-weight)]

#Looping through the rest of the close price data and calculating the EMA at each step and appending it to the list of EMA values
	for i in range(period+1,len(stock)):
		EMA_list.append(stock[i][-1]*weight + EMA_list[-1]*(1-weight))
	return np.array(EMA_list)

EMA = EMA_func(smoothing_factor, MA_size, jnj_prices)




#MACD

#Calculating the MACD and making sure that the EMA12 and EMA 26 have the same number of data points
def MACD_func(smoothing, stock):
	return np.subtract(EMA_func(smoothing, 12, stock)[26-12:], EMA_func(smoothing, 26, stock))

MACD = MACD_func(smoothing_factor, jnj_prices)



#RSI 14-Day

def RSI_func(period, stock):

#Holds the average gain over the period of the RSI
	avg_gain = 0
#Holds the average loss over the period of the RSI
	avg_loss = 0

#Calculate the first value of the RSI using the average gain and loss values over the period
	for i in range(1,period+1):
# Take the difference between adjacent close price values
		diff = stock[i][-1] - stock[i-1][-1]
#if it is a gain add it to the average gain
		if diff >= 0:
			avg_gain += diff
#if it is a loss add it to the average loss
		else:
			avg_loss += diff

#Calculate the average gain and loss by dividing by the period
	avg_gain = avg_gain/period

	avg_loss = abs(avg_loss/period)

#Creating a container to store the average gain and loss for each data point
	avg_gain_list = [avg_gain]

	avg_loss_list = [avg_loss]

#Calculate the first RSI value and store it in a list so future values can be appended
	RSI_list = [100 - (100/(1+(avg_gain/avg_loss)))]

#Loop over the remaining data points in the close price
	for i in range(period+1,len(stock)):
		diff = stock[i][-1] - stock[i-1][-1]
		if diff >= 0:
#Calculate the average gain and RSI
			avg_gain = (avg_gain*(period-1) + diff)/period
			RSI_list.append(100 - (100/(1+(avg_gain/avg_loss))))
		else:
#Calculate the average gain and RSI
			avg_loss = (avg_loss*(period-1) + abs(diff))/period
			RSI_list.append(100 - (100/(1+(avg_gain/avg_loss))))

		avg_gain_list.append(avg_gain)
		avg_loss_list.append(avg_loss)

	return avg_gain_list, avg_loss_list, np.array(RSI_list)


avg_gain, avg_loss, RSI = RSI_func(14, jnj_prices)
	


#Copy all of the technical indicator data and adjust it so that they are all of the same length and begin at the same date, since MACD is the longest use it to adjust indicator list lengths
SMA_adj = copy.deepcopy(SMA[len(SMA)-len(MACD):])
EMA_adj = copy.deepcopy(EMA[len(EMA)-len(MACD):])
MACD_adj = copy.deepcopy(MACD)
RSI_adj = copy.deepcopy(RSI[len(RSI)-len(MACD):])
jnj_prices_adj = copy.deepcopy(jnj_prices[len(jnj_prices)-len(MACD):])


#Add all of the technical indicators to the price data list
for i in range(0,len(jnj_prices_adj)):
	jnj_prices_adj[i].extend([SMA_adj[i], EMA_adj[i], MACD_adj[i], RSI_adj[i]])

#Adjust the date data so that it is the same length as the price data
jnj_list = copy.deepcopy(jnj_date)
jnj_list = jnj_list[len(jnj_list)-len(MACD):]

#Calculate the first time difference between adjacent dates
time_diff_1 = jnj_date[len(jnj_date)-len(MACD):len(jnj_date)-len(MACD)+1][0][0] - jnj_date[len(jnj_date)-len(MACD)-1:len(jnj_date)-len(MACD)][0][0] 

#Store the first time difference in a list
jnj_time_diff = [time_diff_1]

#Calculate and store the time difference for the rest of the dates
for i in range(1, len(jnj_list)):
	jnj_time_diff.append(jnj_list[i][0] - jnj_list[i-1][0])

#Add all of the date data, time differences and price data into a single list
for i in range(0,len(jnj_prices_adj)):
	jnj_list[i].extend([jnj_time_diff[i]])
	jnj_list[i].extend(jnj_prices_adj[i])
	

# Turn the list with all of the data into a pandas dataframe
jnj = pd.DataFrame(jnj_list, columns = ['time','year','season','month','week','weekday','time diff','volume','open','high','low','close','SMA','EMA','MACD','RSI']).set_index('time')


#Calculates the sine and cosine transformed cyclical features and inserts them into the dataframe
def cyclical(stock, time_values, max_value):
	stock.insert(stock.columns.get_loc(time_values)+1, time_values + ' cos', np.cos(2 * np.pi * stock[time_values].values/max_value))
	stock.insert(stock.columns.get_loc(time_values)+1, time_values + ' sin', np.sin(2 * np.pi * stock[time_values].values/max_value))
	return stock

#Calculating the sine and cosine transforms for each cyclical feature in the dataset
jnj = cyclical(jnj,'season', jnj['season'].max())
jnj = cyclical(jnj,'month', jnj['month'].max())
jnj = cyclical(jnj,'week', jnj['week'].max())
jnj = cyclical(jnj,'weekday', jnj['weekday'].max())


#Splits the data into training, test and validation sets as well as what features want to be included and the size of the sliding window for the close price
def windowing(num_lookback, stock_DataFrame, train_size_decimal, validation_size_decimal, fin_features):

	windowed_stock_DataFrame = copy.deepcopy(stock_DataFrame)

#How large of a sliding window do we want
	windows = range(1,num_lookback+1)
	
#Insert the previous close price for the size of the sliding window as new columns in the dataframe
	for window in windows:
		windowed_stock_DataFrame.insert(loc = len(stock_DataFrame.columns)-1+window,column='close-{}'.format(window),value = windowed_stock_DataFrame['close'].shift(window))
	
	windowed_stock_DataFrame = windowed_stock_DataFrame.drop(index=windowed_stock_DataFrame.index[:num_lookback])

#Size of the training set
	train_split = int(train_size_decimal*len(windowed_stock_DataFrame.index))

#Size of the  validation set
	val_split = int((train_size_decimal+validation_size_decimal)*len(windowed_stock_DataFrame.index))

	df_x = copy.deepcopy(windowed_stock_DataFrame)

#If we dont want the features engineered from financial data drop those columns as well as the non cyclical date columns and close price and other features that have look ahead bias
	if fin_features == False:
		df_x = df_x.drop(columns = ['season','month','week','weekday','volume','open','high','low','close','SMA','EMA','MACD','RSI'])
		for window in windows:
			df_x = df_x.drop(columns = ['close-{}'.format(window)])
	else:
#drop the non cyclical date columns and close price and other features that have look ahead bias
		df_x = df_x.drop(columns = ['season','month','week','weekday','volume','open','high','low','close'])

#Make the training, validation and test set feature dataframes
	df_x_train = df_x.iloc[:train_split]

	df_x_val = df_x.iloc[train_split:val_split]

	df_x_test = df_x.iloc[val_split:]


#mke the training, validation and test set for the target variable dataframes and rename feature dataframes for consistent naming
	x_train_in = df_x_train 

	y_train_in = windowed_stock_DataFrame.iloc[:train_split][['close']] 

	x_val_in = df_x_val

	y_val_in = windowed_stock_DataFrame.iloc[train_split:val_split][['close']]

	x_test_in = df_x_test 

	y_test_in = windowed_stock_DataFrame.iloc[val_split:][['close']] 


	return x_train_in, y_train_in, x_val_in, y_val_in, x_test_in, y_test_in


#If we want to include features engineered from financial data in the model
Financial_Features = False

#If we want to predict the close price on January 1, 2022. Only works when not using features engineered from financial data.
jan_1_2022_predict = False

# Perform a grid search to find the best parameters for the gradient boosted trees model
Grid_Search = False

#Plot the feature importance plot for the model
Plot_Feature_Importance = False

#Do we only want to use data in reference to when the Fed began quantitative easing in March 2020.
Quant_Ease_Date = True

#Do we want to use the data after quantitative easing began.
Quant_Ease_Post = True

#Splits the dataframe to only use data before or after quantitative easing began.
if Quant_Ease_Date == True:
	if Quant_Ease_Post == True:
		jnj = jnj.iloc[jnj.index.get_loc(datetime.strptime('2020-03-15', '%Y-%m-%d').timestamp(), method = 'nearest'):]
	else:
		jnj = jnj.iloc[:jnj.index.get_loc(datetime.strptime('2020-03-15', '%Y-%m-%d').timestamp(), method = 'nearest')]

#How big of a sliding window do you want, how many previous close prices do you want as features
lookback = 7

#Size of the training set
train_size = 0.50

#Size of the validation set
validation_size = 0.20

#Calling the function that gets the data frames for each set
x_train, y_train, x_val, y_val, x_test, y_test = windowing(lookback, jnj, train_size, validation_size, Financial_Features)


#Getting a list of the time from the training, validation and test set for plotting 
x_train_time = []
for time_stamp in list(x_train.index.values):
	x_train_time.append(datetime.fromtimestamp(time_stamp))

x_val_time = []
for time_stamp in list(x_val.index.values):
	x_val_time.append(datetime.fromtimestamp(time_stamp))

x_test_time = []
for time_stamp in list(x_test.index.values):
	x_test_time.append(datetime.fromtimestamp(time_stamp))


#Adding the data for January 1st 2022 to the data frames
if jan_1_2022_predict == True:
	if Financial_Features == False:
		#Getting the feature data
		x_test = pd.concat([x_test, pd.DataFrame([[2022,np.sin(2 * np.pi * 1/4), np.cos(2 * np.pi * 1/4), np.sin(2 * np.pi * 1/12), np.cos(2 * np.pi * 1/12), np.sin(2 * np.pi * 1/52), np.cos(2 * np.pi * 1/52), np.sin(2 * np.pi * 0/6), np.cos(2 * np.pi * 0/6), datetime.strptime('2022-01-01', '%Y-%m-%d').timestamp() - list(x_test.index)[-1]]], columns = list(x_test.columns), index= [datetime.strptime('2022-01-01', '%Y-%m-%d').timestamp()])])
		#Getting the time for plotting
		x_test_time.append(datetime.fromtimestamp(datetime.strptime('2022-01-01', '%Y-%m-%d').timestamp()))
		#Using the last close price in the test set as the close price on Janauary 1, 2022 so that all of the data is the same size
		y_test = pd.concat([y_test, pd.DataFrame([y_test.values[-1]], columns = ['close'], index= [datetime.strptime('2022-01-01', '%Y-%m-%d').timestamp()])])
	else:
		print("Warning: jan_1_2022_predict flag is True but Financial_Features flag is False. The program will not predict the close price for January 1st 2022.")

#Creating a list of the time for all of the data
x_tot_time = copy.deepcopy(x_train_time)
x_tot_time.extend(x_val_time)
x_tot_time.extend(x_test_time)

#Copying the feature and target training data, this will be necessary when predicting prices using the sliding window method
x_pred = copy.deepcopy(x_train)

y_pred = copy.deepcopy(y_train)



#Calculating the EMA20, EMA26, and EMA12. The list of these values will be useful for recalculating the technical indicators in the sliding window
EMA_20 = EMA_func(smoothing_factor, 20, jnj_prices)
EMA_26 = EMA_func(smoothing_factor, 26, jnj_prices)
EMA_12 = EMA_func(smoothing_factor, 12, jnj_prices)

EMA_20_list = EMA_20[len(EMA_20)-len(MACD):][lookback:][-(len(x_val)+1):-len(x_val)]
EMA_26_list = EMA_26[len(EMA_26)-len(MACD):][lookback:][-(len(x_val)+1):-len(x_val)]
EMA_12_list = EMA_12[len(EMA_12)-len(MACD):][lookback:][-(len(x_val)+1):-len(x_val)]

#The EMA calculation function but adjusted for use in the sliding window to avoid look ahead bias
def EMA_test(smoothing, period, test_set_price, EMA_set):

	weight = smoothing/(period+1)
	
	return test_set_price*weight + EMA_set[-1]*(1-weight)


#RSI 14-Day
#The RSI calculation function but adjusted for use in the sliding window to avoid look ahead bias
def RSI_test(period, test_set_price_prev,test_set_price_prev_2, gain, loss):

	diff = test_set_price_prev - test_set_price_prev_2
	if diff >= 0:
		gain = (gain*(period-1) + diff)/(period)
		RSI_curr = 100 - (100/(1+(gain/loss)))
	else:
		loss = (loss*(period-1) + abs(diff))/(period)
		RSI_curr = 100 - (100/(1+(gain/loss)))

	return gain, loss, RSI_curr


#avg gain and loss at the end of training set
avg_gain_test = avg_gain[len(avg_gain)-len(MACD):][lookback:][-(len(x_val)+1):-len(x_val)][0]
avg_loss_test = avg_loss[len(avg_loss)-len(MACD):][lookback:][-(len(x_val)+1):-len(x_val)][0]





#No Technical Indicators
#with year and indicators {'gamma': 0.001, 'learning_rate': 0.01, 'max_depth': 12, 'random_state': 42} #400 or {'gamma': 0.04, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 600, , 'random_state': 42},

#without year or indicatores {'gamma' : 0.005, 'learning_rate' : 0.05, 'max_depth' : 8, 'random_state' : 42} #200

#without year or times or indicators {'gamma': 0.02, 'learning_rate': 0.01, 'max_depth': 8, 'random_state': 42} #400

#If you want to include the features engineered from financial data use this method
if Financial_Features == True:

	#Parameters for the gradient boosted trees model
	param = {'random_state': 42}

	#The number of decision trees to be used in the model
	num_round = 100

	#First predict the values of the validation set and then the test set
	for x_future in [x_val, x_test]:


		# Set to zero to make sure there is no look ahead bias
		x_future['SMA'] = 0.0
		x_future['EMA'] = 0.0
		x_future['MACD'] = 0.0
		x_future['RSI'] = 0.0
		
		for i in range(1,lookback+1):
			x_future['close-{}'.format(i)] = 0.0



		#looping through all of the values in the validation or test set
		for i in range(0,len(list(x_future.index.values))):

			#Setting all of the previous close price features from the previous close price target variables in the current iteration
			for j in range(1,lookback+1):
				x_future.iat[i,x_future.columns.get_loc('close-{}'.format(j))] = y_pred['close'].values[-j]

			#Calculating the SMA20 for the current iteration
			x_future.iat[i,x_future.columns.get_loc('SMA')] = sum(y_pred['close'].values[-20:])/20

			#Calculating the EMA20 for the current iteration
			x_future.iat[i,x_future.columns.get_loc('EMA')] = EMA_test(smoothing_factor,20, y_pred['close'].values[-1], EMA_20_list)

			#Appending to the list of EMA20 values so that the value of the current iteration can be used during the next iteration
			np.append(EMA_20_list,EMA_test(smoothing_factor,20, y_pred['close'].values[-1], EMA_20_list))

			#Calculating the MACD for the current iteration
			x_future.iat[i,x_future.columns.get_loc('MACD')] = EMA_test(smoothing_factor,12, y_pred['close'].values[-1], EMA_12_list) - EMA_test(smoothing_factor,26, y_pred['close'].values[-1], EMA_26_list)

			#Appending to the list of EMA12 and EMA 26 values so that the value of the current iteration can be used during the MACD calculation in the next iteration
			np.append(EMA_12_list,EMA_test(smoothing_factor,12, y_pred['close'].values[-1], EMA_12_list))

			np.append(EMA_26_list,EMA_test(smoothing_factor,26, y_pred['close'].values[-1], EMA_26_list))

			#Calculating RSI and the average gain and loss so that they can be used in the next iteration
			avg_gain_test, avg_loss_test, x_future.iat[i,x_future.columns.get_loc('RSI')]  = RSI_test(14, y_pred['close'].values[-1], y_pred['close'].values[-2], avg_gain_test, avg_loss_test)

			#Using all of the feature data up to the current iteration to train the model
			grad_boost_reg = xgb.train(param,xgb.DMatrix(x_pred.values,y_pred.values),num_round)

			#Predicting the close price for one time step in the future
			y_pred = y_pred.append(pd.DataFrame(grad_boost_reg.predict(xgb.DMatrix(np.array(x_future.iloc[i].values).reshape(1,-1))), columns=['close'], index=[x_future.index[i]]))

			#Adding all of the financial features calculated during the current iteration to the training data set
			x_pred = x_pred.append(pd.DataFrame(np.array(x_future.iloc[i].values).reshape(1,-1), columns=list(x_pred.columns.values), index=[x_future.index[i]]))



		#Deciding if a feature importance plot wants to be plotted on the data set with the financial features
		if Plot_Feature_Importance == True:	
			model = xgb.XGBRegressor(random_state = param['random_state'])#xgb.XGBRegressor(gamma = param['gamma'], learning_rate = param['learning_rate'], max_depth= param['max_depth'], n_estimators = num_round, random_state = param['random_state'])
			model.fit(x_train, y_train)
			model.predict(np.array(x_val.head(1).values))
			fig, ax = plt.subplots(1,1)
			plot_importance(model, ax=ax)
			ax.set_title('Feature Importantce',fontsize=20)
			ax.set_xlabel('F-Score',fontsize=16)
			ax.set_ylabel('Features',fontsize=16)
			#ax.set_xticklabels(ax.get_xticklabels(),fontsize=16)
			ax.set_yticklabels(ax.get_yticklabels(),fontsize=16)
			plt.show()


		#Deciding if a grid search wants to be performed on the data set with the financial features
		if Grid_Search == True:
			#parameter search grid
			parameters = {
		    'n_estimators': [100, 200, 400, 600],
		    'learning_rate': [ 0.005, 0.01, 0.05, 0.1],
		    'max_depth': [3,5,8, 10],
		    'gamma': [0.005, 0.01, 0.02, 0.04],
		    'random_state': [42]
			}
			#Using the training set and the first value in the validation set to fit the grid search parameters to
			eval_set = [(x_train, y_train), (x_val.head(1), y_val.head(1))]
			model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbosity=0)
			clf = GridSearchCV(model, parameters)

			clf.fit(x_train, y_train)

			#Printing the parameters and the corresponding validation score
			print(f'Best params: {clf.best_params_}')
			print(f'Best validation score = {clf.best_score_}')




else:

	#Parameters for the gradient boosted trees model
	param = {'gamma': 0.01, 'learning_rate': 0.05, 'max_depth': 10, 'random_state' : 42}#{'gamma' : 0.005, 'learning_rate' : 0.05, 'max_depth' : 8, 'random_state' : 42} 

	#The number of decision trees to be used in the model
	num_round = 100#200

	#Training the model on the training data
	grad_boost_reg = xgb.train(param,xgb.DMatrix(x_pred.values,y_pred.values),num_round)

	#Predicting the close price values for the validation set
	y_pred = y_pred.append(pd.DataFrame(grad_boost_reg.predict(xgb.DMatrix(x_val.values)), columns=['close'], index=[x_val.index]))

	#Predicting the close price values for the test set
	y_pred = y_pred.append(pd.DataFrame(grad_boost_reg.predict(xgb.DMatrix(x_test.values)), columns=['close'], index=[x_test.index]))


	#Deciding if a feature importance plot wants to be plotted on the data set
	if Plot_Feature_Importance == True:
		model = xgb.XGBRegressor(gamma = param['gamma'], learning_rate = param['learning_rate'], max_depth= param['max_depth'], n_estimators = num_round, random_state = param['random_state'])
		model.fit(x_train, y_train)
		model.predict(np.array(x_val.values))
		fig, ax = plt.subplots(1,1)
		plot_importance(model, ax=ax)
		ax.set_title('Feature Importantce',fontsize=20)
		ax.set_xlabel('F-Score',fontsize=16)
		ax.set_ylabel('Features',fontsize=16)
		#ax.set_xticklabels(ax.get_xticklabels(),fontsize=16)
		ax.set_yticklabels(ax.get_yticklabels(),fontsize=16)
		plt.show()


	#Deciding if a grid search wants to be performed on the data set with the financial features
	if Grid_Search == True:
		parameters = {
	    'n_estimators': [100, 200, 400, 600],
	    'learning_rate': [ 0.005, 0.01, 0.05, 0.1],
	    'max_depth': [3,5,8, 10],
	    'gamma': [0.005, 0.01, 0.02, 0.04],
	    'random_state': [42]
		}

		eval_set = [(x_train, y_train), (x_val, y_val)]
		model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbosity=0)
		clf = GridSearchCV(model, parameters)

		clf.fit(x_train, y_train)

		print(f'Best params: {clf.best_params_}')
		print(f'Best validation score = {clf.best_score_}')


#Getting the maximum and minimum value in the training set so that they can be later plotted as horizontal lines
train_max = y_train.index.get_loc(y_train['close'][y_train['close'] == max(y_train['close'].values)].index[0])
train_min = y_train.index.get_loc(y_train['close'][y_train['close'] == min(y_train['close'].values)].index[0])


#Getting all of the train, validation and test set data into a single dataframe for plotting and calculations
y_all = copy.deepcopy(y_train)
y_all = y_all.append(pd.DataFrame(y_val.values, columns=list(y_val.columns.values), index=[y_val.index]))
y_all = y_all.append(pd.DataFrame(y_test.values, columns=list(y_test.columns.values), index=[y_test.index]))

#Calculating the mean squared error, mean absolute error and mean percentage error for the close price predictions
print('mean squared error = ',mean_squared_error(y_all['close'].values[len(x_train_time):], y_pred['close'].values[len(x_train_time):]))
print('mean absolute error = ',mean_absolute_error(y_all['close'].values[len(x_train_time):], y_pred['close'].values[len(x_train_time):]))
print('mean absolute percentage error = ',mean_absolute_percentage_error(y_all['close'].values[len(x_train_time):], y_pred['close'].values[len(x_train_time):]))

#plotting the training set, the validation set and test set close price with respect to the date as well as the predicted close price on the training and validation set
plt.plot(x_tot_time[len(x_train_time):],y_pred['close'].values[len(x_train_time):], label = 'Prediction')
plt.plot(x_train_time, y_train['close'].values, label = 'Training Set')
plt.plot(x_val_time, y_val['close'].values, label = 'Validation Set')
plt.plot(x_test_time, y_test['close'].values, label = 'Test Set')
#Plotting horizontal lines for the minimum and maximum of the training set to demonstrate the bounds of predictions for the model
plt.hlines(y = max(y_train['close'].values), xmin = x_train_time[int(train_max)], xmax = x_tot_time[-1], linestyle ='--',color = 'purple', label = 'Decision Tree Bounds')
plt.hlines(y = min(y_train['close'].values), xmin = x_train_time[int(train_min)], xmax = x_tot_time[-1], linestyle ='--',color = 'purple')

plt.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize = 20)
plt.xlabel('Time',fontsize = 20)
plt.ylabel('Price, USD',fontsize = 20)
plt.title('Johnson & Johnson Stock Price, Gradient Boosted Trees (XGBoost)', fontsize = 28) #
plt.show()




#Plot for demonstraing the abilities of cyclically encoding features
#fig, axs = plt.subplots(2, 2)
#axs[0, 0].plot(range(0,23), range(0,23))
#axs[0, 0].set_title('Hour')
#axs[0, 1].plot(range(0,23), np.cos(2 * np.pi * np.array(range(0,23))/23), 'tab:orange')
#axs[0, 1].set_title('Hour Cosine')
#axs[1, 0].plot(range(0,23), np.sin(2 * np.pi * np.array(range(0,23))/23), 'tab:green')
#axs[1, 0].set_title('Hour Sine')
#axs[1, 1].plot(np.cos(2 * np.pi * np.array(range(0,23))/23), np.sin(2 * np.pi * np.array(range(0,23))/23), 'tab:red', marker ='o', linestyle = 'None')
#axs[1, 1].set_title('Hour Cosine and Sine')

#for ax in axs.flat:
    #ax.set(xlabel='x-label', ylabel='y-label')

#plt.show()

# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()