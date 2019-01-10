import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
import utils

# fixar random seed para se puder reproduzir os resultados
seed = 9
np.random.seed(seed)

#----------------------Etapa 1------------------------------preparar o dataset

def read_cvs_dataset(ficheiro, col_label):
# ler ficheiro csv para matriz numpy, e separar o label que está em col_label (deve ser a ultima coluna)
	dataset = np.loadtxt(ficheiro, delimiter=",")
	print('Formato do dataset: ',dataset.shape)
	input_attributes = dataset[:,0:col_label]
	output_attributes = dataset[:,col_label]
	print('Formato das variáveis de entrada (input variables):',input_attributes.shape)
	print('Formato da classe de saída (output variables): ',output_attributes.shape)
	#print(X[0])
	#print(Y[0])
	return (input_attributes,output_attributes)

#------------------------Etapa 2--------------------------Definir a topologia da rede (arquitectura do modelo)
'''
cria-se um modelo sequencial e vai-se acrescentando camadas (layers) vamos criar 3 camadas no nosso modelo 
Dense class significa que teremos um modelo fully connected o primeiro parametro estabelece o número de neuronios na camada (12 na primeira)
input_dim=8 indica o número de entradas do nosso dataset (8 atributos neste caso)
kernel_initializer indica o metodo de inicialização dos pesos das ligações
'uniforme' sigifica small random number generator com default entre 0 e 0.05 outra hipotese seria 'normal' com small number generator from Gaussion distribution
"activation" indica a activation fuction
'relu' rectifier linear unit activation function com range entre 0 e infinito
'sigmoid' foi utilizada para garantir um resultado entre 0 e 1
'''
def create_model():
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation="relu", kernel_initializer="uniform"))
	model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
	model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
	return model

#-------------------Etapa 3------------------------Compilar o modelo (especificar o modelo de aprendizagem a ser utilizado pela rede)
'''
loss - funcão a ser utilizada no calculo da diferença entre o pretendido e o obtido 
vamos utilizar logaritmic loss para classificação binária: 'binary_crossentropy'
o algoritmo de gradient descent será o “adam” pois é eficiente a métrica a ser utilizada no report durante o treino será 'accuracy' 
pois trata-se de um problema de classificacao
'''
def compile_model(model):
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

'''
Binary accuracy
K.round(y_pred) implies that the threshold is 0.5,
everything above 0.5 will be considered as correct.
'''
def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
'''
Categorical accuracy
K.argmax(y_true) takes the highest value to be the prediction and matches against the comparative set.
'''
def categorical_accuracy(y_true, y_pred):
	return K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())
'''
Sparse categorical accuracy:
Might be a better metric than categorical_accuracy in some cases depending on your data.
'''
def sparse_categorical_accuracy(y_true, y_pred):
	return K.cast(K.equal(K.max(y_true, axis=-1),K.cast(K.argmax(y_pred, axis=-1), K.floatx())), K.floatx())
'''
Top-k categorical accuracy:
Top-k is measured on the accuracy of the correct prediction being in the top-k predictions.
Most papers will report on how well a models does based on top-5 accuracy.
'''
def top_k_categorical_accuracy(y_true, y_pred, k=5):
	return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)

#----------------Etapa 4---------------treinar a rede (Fit the model) neste caso foi feito com os dados todos
'''
'batch_size'núermo da casos processados de cada vez model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=2)
verbose: 0 para print do log de treino stdout, 1 para barra de progresso, 2 para uma linha por epoch.
validation_split: float (0. < x < 1). Fração de dados a serem utilizados como dados de validação.
'''
def fit_model(model,input_attributes,output_attributes):
	history = model.fit(input_attributes, output_attributes, validation_split=0.33,
	epochs=150, batch_size=10, verbose=1)
	return history

#------------------Etapa 5--------------Calcular o desempenho do modelo treinado (neste caso utilizando os dados usados no treino)
def model_evaluate(model,input_attributes,output_attributes):
	print("###########inicio do evaluate###############################\n")
	scores = model.evaluate(input_attributes, output_attributes)
	print("\n metrica: %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))

#-----------------Etapa 6-------------------Utilizar o modelo treinado e escrever as previsões para novos casos
def model_print_predictions(model,input_attributes,output_attributes):
	previsoes = model.predict(input_attributes)
	# arredondar para 0 ou 1 pois pretende-se um output binário
	LP=[]
	for prev in previsoes:
		LP.append(round(prev[0]))
	#LP = [round(prev[0]) for prev in previsoes]
	for i in range(len(output_attributes)):
		print(" Class:",output_attributes[i]," previsão:",LP[i])
		if i>10: break

# Ciclo completo executando as Etapas 1,2,3,4,5 e 6
def ciclo_completo():
	(input_attributes,output_attributes) = read_cvs_dataset("pima-indians-diabetes.csv", 8)
	model = create_model()
	utils.print_model(model,"model_MLP.png")
	compile_model(model)
	history=fit_model(model,input_attributes,output_attributes)
	utils.print_history_loss(history)
	model_evaluate(model,input_attributes,output_attributes)
	model_print_predictions(model,input_attributes,output_attributes)

# exemplos de utilização destes utilitários
def ciclo_ler_dataset_treinar_gravar():
	(input_attributes,output_attributes) = read_cvs_dataset("pima-indians-diabetes.csv",8)
	model = create_model()
	utils.print_model(model,"model2.png")
	compile_model(model)
	history = fit_model(model,input_attributes,output_attributes)
	utils.print_history_accuracy(history)
	utils.print_history_loss(history)
	model_evaluate(model,input_attributes,output_attributes)
	utils.save_model_json(model,"model.json")
	utils.save_weights_hdf5(model,"model.h5")
	return (input_attributes,output_attributes)

def ciclo_ler_modelo_evaluate_usar(input_attributes,output_attributes):
	model = utils.load_model_json("model.json")
	utils.load_weights_hdf5(model,"model.h5")
	compile_model(model)
	model_evaluate(model,input_attributes,output_attributes)
	model_print_predictions(model,input_attributes,output_attributes)

if __name__ == '__main__':
#opção 1 - ciclo completo
	#ciclo_completo()
#opção 2 - ler,treinar o dataset e gravar. Depois ler o modelo e pesos e usar
	(input_attributes,output_attributes) = ciclo_ler_dataset_treinar_gravar()
	ciclo_ler_modelo_evaluate_usar(input_attributes,output_attributes)