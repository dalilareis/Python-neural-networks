import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
import utils 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.preprocessing.data import QuantileTransformer

seed = 9
np.random.seed(seed)


def get_data(file_name='sales.csv', normalizer=None):
	col_names = ['Date','Advertising','Sales', 'Season'] #Season added in the csv file (pre-processing)
	sales = pd.read_csv(file_name, header=0, names=col_names) 
	df = pd.DataFrame(sales)
	date_split = df['Date'].str.split('-').str
	df['Year'], df['Month'] = date_split
	vendas = df['Sales']
	ads = df['Advertising']

	# Get Month as new columns (binarized) ---> 13 columns because function assumes there's a label 0
	cat_Months = np_utils.to_categorical (df['Month'])
	df_Month = pd.DataFrame(cat_Months)
	df_Month.drop(df_Month.columns[[0]], axis=1, inplace=True) #drop column that corresponds to month 0 ---> result: 12 columns
	
	#Categorize Seasons (Seasons & Months are not to be multiplied, so they need to be categorical)
	cat_Seasons = np_utils.to_categorical (df['Season'])
	df_Season = pd.DataFrame(cat_Seasons) # 4 columns (season started in 0)

	# Remove unwanted columns
	df.drop(df.columns[[0, 3, 4, 5]], axis=1, inplace=True) #Keep only Sales and Advertising (2 columns)
	
	# Combine all columns
	df = pd.concat([df, df_Month], axis=1)
	df = pd.concat([df, df_Season], axis=1) #----> TOTAL: 18 Columns

	print(df.head())

	if normalizer is not None:
		df.drop(df.columns[[0, 1]], axis=1, inplace=True) #Remove Sales and Advertising	
		vendas = vendas.values.reshape(-1, 1).astype('float32')
		ads = ads.values.reshape(-1, 1).astype('float32')
		scalerSales = normalizer().fit(vendas)
		vendasNorm = scalerSales.transform(vendas)
		scalerAds = normalizer().fit(ads)
		adsNorm = scalerAds.transform(ads)
		df['Advertising'] = adsNorm
		df['Sales'] = vendasNorm
		
	label = df.pop('Sales') # Move 'Sales' column to end (label)
	df['Sales'] = label

	# print(df.head())
	# print(df.shape)

	return df, scalerSales

def split_data(df_dados, janela):
	qt_atributos = len(df_dados.columns)
	mat_dados = df_dados.as_matrix() #converter dataframe para matriz (lista com lista de cada registo)
	tam_sequencia = janela + 1
	res = []
	for i in range(len(mat_dados) - tam_sequencia): #numero de registos - tamanho da sequencia
		res.append(mat_dados[i: i + tam_sequencia])
	res = np.array(res) #dá como resultado um np com uma lista de matrizes (janela deslizante ao longo da serie)
	qt_casos_treino = int(round((2/3) * res.shape[0])) #2/3 passam a ser casos de treino (primeiros 2 anos)
	train = res[:qt_casos_treino, :]
	x_train = train[:, :-1] #menos um registo pois o ultimo registo é o registo a seguir à janela
	y_train = train[:, -1][:,-1] #para ir buscar o último atributo para a lista dos labels
	x_test = res[qt_casos_treino:, :-1]
	y_test = res[qt_casos_treino:, -1][:,-1]
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], qt_atributos))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], qt_atributos))
	return [x_train, y_train, x_test, y_test]

# Etapa 2 - Definir a topologia da rede (arquitectura do modelo) e compilar '''
def build_model(janela):
	model = Sequential()
	model.add(LSTM(256, input_shape=(janela, 18), return_sequences=True))
	model.add(BatchNormalization())
	model.add(LSTM(128, input_shape=(janela, 18), return_sequences=True)) 
	model.add(BatchNormalization())
	model.add(LSTM(64, input_shape=(janela, 18), return_sequences=True)) 
	model.add(BatchNormalization())
	model.add(LSTM(32, input_shape=(janela, 18), return_sequences=False)) #False pq n há + camadas LSTM a seguir
	model.add(Dropout(0.3))	
	model.add(BatchNormalization())
	model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
	model.add(BatchNormalization())
	model.add(Dense(8, activation="sigmoid", kernel_initializer="uniform"))
	model.add(BatchNormalization())
	model.add(Dense(4, activation="relu", kernel_initializer="uniform"))
	model.add(BatchNormalization())
	model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
	model.add(BatchNormalization())
	model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	return model

#imprime um grafico com os valores de teste e com as correspondentes tabela de previsões
def print_series_prediction(y_test,predic, normalizer=None):
	if normalizer is not None:
		y_test = y_test.reshape(-1, 1).astype('float32')
		y_test = normalizer.inverse_transform(y_test)
		predic = predic.reshape(-1, 1).astype('float32')
		predic = normalizer.inverse_transform(predic)

	diff=[]
	racio=[]
	for i in range(len(y_test)): #para imprimir tabela de previsoes
		racio.append( (y_test[i]/predic[i])-1)
		diff.append( abs(y_test[i]- predic[i]))
		print('valor: %f ---> Previsão: %f Diff: %f Racio: %f' % (y_test[i],predic[i], diff[i], racio[i]))
	plt.plot(y_test,color='blue', label='y_test')
	plt.plot(predic,color='red', label='prediction') #este deu uma linha em branco
	plt.plot(diff,color='gold', label='diff')
	plt.plot(racio,color='olive', label='racio')
	plt.legend(loc='best')
	plt.title('Prediction Curve for Scaled Data')
	if normalizer is not None:
		plt.title('Prediction Curve for Unscaled Data')
	plt.show()

def LSTM_sales_data(normalizer=None):
	df, scaler = get_data(normalizer=normalizer)
	print("Dataset: ", df.shape)

	janela = 6 #tamanho da Janela deslizante (trimestral, mensal, semestral)
	X_train, y_train, X_test, y_test = split_data(df, janela)
	print("X_train", X_train.shape)
	print("y_train", y_train.shape)
	print("X_test", X_test.shape)
	print("y_test", y_test.shape)

	model = build_model(janela)

	model.fit(X_train, y_train, batch_size=10, epochs=300, validation_split=0.1, verbose=1) #validation 0.1 dos 0.66 usados para treino
	
	utils.print_model(model,"lstm_model.png")

	trainScore = model.evaluate(X_train, y_train, verbose=0)
	print('\n Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
	testScore = model.evaluate(X_test, y_test, verbose=0)
	print(' Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
	print('\n****************** UNSCALED*******************')

	# Unscale Results to get real value predictions and error
	trainScore = trainScore[0].reshape(-1, 1).astype('float32')
	unscaled_Train = scaler.inverse_transform(trainScore)
	print('\n Unscaled Train Score: %.2f MSE (%.2f RMSE)' % (unscaled_Train, math.sqrt(unscaled_Train)))
	testScore = testScore[0].reshape(-1, 1).astype('float32')
	unscaled_Test = scaler.inverse_transform(testScore)
	print(' Unscaled Test Score: %.2f MSE (%.2f RMSE) \n' % (unscaled_Test, math.sqrt(unscaled_Test)))

	p = model.predict(X_test)
	predic = np.squeeze(np.asarray(p)) #para transformar uma matriz de uma coluna e n linhas em	um np array de n elementos
	
	print_series_prediction(y_test,predic)
	print('')
	print_series_prediction(y_test,predic, normalizer=scaler)

'''‘ MSE- (Mean square error), RMSE- (root mean square error) – o significado de RMSE depende do range da label. para o mesmo range menor é melhor.
'''
if __name__ == '__main__':

	scalers = ['StandardScaler', 'MinMaxScaler', 'QuantileTransformer']
	
	#get_data(normalizer=QuantileTransformer)
	LSTM_sales_data(normalizer=QuantileTransformer)
	