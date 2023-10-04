from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Datos de entrenamiento
frases = ['Hola, ¿cómo estás?', 'Buenos días', 'Buenas noches', '¿Cómo te llamas?']
etiquetas = ['saludo', 'saludo', 'despedida', 'pregunta']

# Vectorizar las frases
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases)

# Entrenar el clasificador
clf = MultinomialNB()
clf.fit(X, etiquetas)

# Clasificar una nueva frase
nueva_frase = vectorizer.transform(['Buenas noches'])
print(clf.predict(nueva_frase))  # Salida: ['despedida']
class InteligenciaArtificial:
    def __init__(self):
        self.conocimiento = {}

    def aprender(self, clave, valor):
        self.conocimiento[clave] = valor

    def responder(self, pregunta):
        if pregunta in self.conocimiento:
            return self.conocimiento[pregunta]
        else:
            return "Lo siento, no tengo información sobre eso."

# Crear una instancia de la IA
mi_ia = InteligenciaArtificial()

# Enseñar a la IA
mi_ia.aprender("saludo", "¡Hola, cómo puedo ayudarte hoy!")

# Obtener una respuesta de la IA
respuesta = mi_ia.responder("saludo")
print(respuesta)
pip install chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Crear una nueva instancia de un ChatBot
mi_bot = ChatBot('MiBot')

# Crear un nuevo entrenador para el chatbot
entrenador = ChatterBotCorpusTrainer(mi_bot)

# Entrenar el chatbot con el corpus de datos en inglés
entrenador.train("chatterbot.corpus.english")

# Obtener una respuesta a una entrada del usuario
respuesta = mi_bot.get_response("¿Cómo estás?")
print(respuesta)
# Importando las bibliotecas necesarias
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd

# Cargando los datos
datos = pd.read_csv('ruta/a/tu/archivo.csv') # Asegúrate de que la ruta al archivo csv es correcta

# Definiendo las variables dependiente e independiente
X = datos['variable_independiente'].values.reshape(-1,1)
y = datos['variable_dependiente'].values.reshape(-1,1)

# Dividiendo los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creando el objeto de regresión lineal
regressor = LinearRegression()  

# Entrenando el algoritmo
regressor.fit(X_train, y_train)

# Haciendo predicciones
y_pred = regressor.predict(X_test)
# Importando las bibliotecas necesarias
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# Cargando los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizando los valores de los píxeles a estar entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Creando el modelo de la red neuronal
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10)
])

# Compilando el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenando el modelo
model.fit(x_train, y_train, epochs=5)

# Evaluando el modelo
model.evaluate(x_test,  y_test, verbose=2)
