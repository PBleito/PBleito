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
