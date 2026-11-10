# Procesamiento de Lenguaje Natural I

## Desafío 1

Notebook: [Desafio_1.ipynb](Desafio_1.ipynb)

Vectorización de texto con TF-IDF y clasificación con Naïve Bayes sobre el dataset **20 Newsgroups** (~18 000 posts en 20 temas, cargado con `sklearn.datasets.fetch_20newsgroups`, sin headers, footers ni quotes).

### Contenido

1. **Similaridad de documentos**: se eligen 5 documentos al azar (semilla fija) y se buscan sus 5 vecinos más similares por similaridad coseno sobre TF-IDF. Después se analiza si los vecinos tienen sentido por contenido y por etiqueta.
2. **Clasificador por prototipos (zero-shot / 1-NN)**: a cada documento de test se le asigna la clase del documento de train más similar.
3. **Optimización de Naïve Bayes**: búsqueda de hiperparámetros del vectorizador (`stop_words`, `sublinear_tf`, `max_df`, `min_df`, `use_idf`, `norm`) y del modelo (`MultinomialNB` / `ComplementNB`, `alpha`, `fit_prior`) para maximizar el F1 macro en test, sin modificar `ngram_range`.
4. **Similaridad entre palabras**: se transpone la matriz documento-término para obtener vectores de palabras y se buscan los 5 términos más similares a *car*, *god*, *space*, *gun* y *windows*.

### Requisitos

```bash
pip install numpy scikit-learn pandas
```

El dataset se descarga automáticamente la primera vez que se ejecuta `fetch_20newsgroups`.

## Desafío 2

Notebook: [Desafio_2.ipynb](Desafio_2.ipynb)

Embeddings propios con Word2Vec (Gensim) entrenados sobre el texto completo de *Don Quijote de la Mancha* (Project Gutenberg, ~380 000 palabras).

### Contenido

1. **Preprocesamiento**: el texto se divide en párrafos y oraciones, se pasa a minúsculas y se tokeniza con una expresión regular.
2. **Entrenamiento**: skip-gram con `vector_size=100`, `window=5`, `min_count=5`, `negative=10` y 30 épocas, con seguimiento de la pérdida por época.
3. **Términos más y menos similares**: se analizan diez términos de interés, pares de palabras, analogías y `doesnt_match`.
4. **Skip-gram vs. CBOW**: se comparan los vecinos que obtiene cada arquitectura.
5. **Visualización**: proyección t-SNE a 2D de las 150 palabras más frecuentes sin stopwords e interpretación de los grupos observados. Las capturas se guardan en `img/`.

### Requisitos

```bash
pip install numpy pandas matplotlib gensim scikit-learn spacy plotly nbformat
```

El corpus se descarga automáticamente en `data/` la primera vez que se ejecuta el notebook.
