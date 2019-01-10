import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
import matplotlib.pyplot as plt

#util para visualizar a topologia da rede num ficheiro em pdf ou png
def print_model(model,fich):
	from keras.utils import plot_model
	plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)

#utils para visulaização do historial de aprendizagem
def print_history_accuracy(history):
	print(history.history.keys())
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def print_history_loss(history):
	print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

# Utils para gravar modelos e pesos utilizá-los posteriormente
'''
Gravar um modelo num ficheiro utilizando o formato json. O nome do ficheiro deve ter a extensão .json
'''
def save_model_json(model,fich):
	model_json = model.to_json()
	with open(fich, "w") as json_file:
		json_file.write(model_json)
'''
Gravar um modelo num ficheiro utilizando o formato yaml. O nome do ficheiro deve ter a extensão .yaml
'''
def save_model_yaml(model,fich):
	model_yaml = model.to_yaml()
	with open(fich, "w") as yaml_file:
		yaml_file.write(model_yaml)
'''
Gravar os pesos de um modelo treinado num ficheiro utilizando o formato HDF5. O nome do ficheiro deve ter a extensão .h5
'''
def save_weights_hdf5(model,fich):
	model.save_weights(fich)
	print("Saved model to disk")

'''
Ler um modelo de um ficheiro no formato json e criar o respetivo modelo em memória.
'''
def load_model_json(fich):
	json_file = open(fich, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	return loaded_model
'''
Ler um modelo de um ficheiro no formato yaml e criar o respetivo modelo em memória.
'''
def load_model_yaml(fich):
	yaml_file = open(fich, 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	return model_from_yaml(loaded_model_yaml)
'''
Ler os pesos um modelo treinado de um ficheiro no formato hdf5 para o respectivo
modelo.
'''
def load_weights_hdf5(model,fich):
	model.load_weights(fich)
	print("Loaded model from disk")