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
