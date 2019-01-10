import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import np_utils
import matplotlib.pyplot as plt
import utils

# fixar random seed para se poder reproduzir os resultados
seed = 9
np.random.seed(seed)

#----------------------------------------------------Etapa 1 - preparar o dataset------------------------------------------------------------------------
'''
fazer o download do MNIST dataset com imagens de digitos escritos à mão para fazer a sua classificação (já pré-preparados)
dataset: https://s3.amazonaws.com/img-datasets/mnist.npz
O ficheiro já tem tudo separado nos ficheiros {x_test.npy, x_train.npy, y_test.npy, y_train.npy}
Os atributos de entrada estão com matrizes 3D(imagem, largura,altura) e os atributos de saída é uma lista com o número correspondente
'''
def load_mnist_dataset(path):
	f = np.load(path)
	x_train = f['x_train']
	y_train = f['y_train']
	x_test = f['x_test']
	y_test = f['y_test']
	f.close()
	return (x_train, y_train), (x_test, y_test)

# Visualizar 6 imagens do mnist numa escala de cinzentos
def visualize_mnist():
	(X_train, y_train), (X_test, y_test) = load_mnist_dataset('mnist.npz')
	plt.subplot(321)
	plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
	plt.subplot(322)
	plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
	plt.subplot(323)
	plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
	plt.subplot(324)
	plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
	plt.subplot(325)
	plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))
	plt.subplot(326)
	plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))
	plt.show()

#----------------------------Etapa 2 - Definir a topologia da rede (arquitectura do modelo) e compilar (multilayer_perceptrons)----------------------------------

def create_compile_model_mlp(num_pixels, num_classes):
	model = Sequential()
	model.add(Dropout(0.2, input_shape=(num_pixels, )))
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	#model.add(BatchNormalization())
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def mnist_utilizando_mlp():
	(X_train, y_train), (X_test, y_test) = load_mnist_dataset('mnist.npz')

	# transformar a matriz 28*28 das imagens num vector com 784 atributos para cada imagem (porque é multilayer-perceptron)
	num_pixels = X_train.shape[1] * X_train.shape[2]
	X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
	X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

	# normalizar os valores dos pixeis de 0-255 para 0-1
	X_train = X_train / 255
	X_test = X_test / 255

	# transformar o label que é um inteiro em categorias binárias, o valor passa a ser o correspondente à posição
	# o 5 passa a ser a lista [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	num_classes = y_test.shape[1]

	# definir a topologia da rede e compilar
	model = create_compile_model_mlp(num_pixels, num_classes)
	utils.print_model(model,"model.png")

	# treinar a rede
	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=50, verbose=2)
	#utils.print_history_accuracy(history)
	utils.print_history_loss(history)

	# Avaliação final com os casos de teste
	scores = model.evaluate(X_test, y_test, verbose = 0)
	print('Scores: ', scores)
	print("Erro modelo MLP: %.2f%%" % (100-scores[1]*100))

#--------------------------------------------------------------MAIN--------------------------------------------------------------------------
if __name__ == '__main__':
	#visualize_mnist()
	mnist_utilizando_mlp()
