from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

embedder = FaceNet()
facenet = embedder.model

# Congelar todas las capas menos las últimas 10
for layer in facenet.layers[:-10]:
    layer.trainable = False
for layer in facenet.layers[-10:]:
    layer.trainable = True

# Definir la capa de pérdida tripleta
class TripletLossLayer(Layer):
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        anchor, positive, negative = inputs
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + self.alpha, 0.0)
        self.add_loss(tf.reduce_mean(loss))
        return inputs

# Crear las entradas para el modelo de entrenamiento tripleta
entrada_ancla = Input(shape=(160, 160, 3), name='entrada_ancla')
entrada_positiva = Input(shape=(160, 160, 3), name='entrada_positiva')
entrada_negativa = Input(shape=(160, 160, 3), name='entrada_negativa')

# Obtener las incrustaciones de las imágenes de entrada
embed_ancla = facenet(entrada_ancla)
embed_positiva = facenet(entrada_positiva)
embed_negativa = facenet(entrada_negativa)

# Crear la capa de pérdida tripleta
loss_out = TripletLossLayer(alpha=0.2)([embed_ancla, embed_positiva, embed_negativa])

# Crear el modelo de entrenamiento tripleta
triplet_model = Model(
    inputs=[entrada_ancla, entrada_positiva, entrada_negativa],
    outputs=loss_out
)

# Compilar el modelo con un optimizador y una tasa de aprendizaje ajustada
triplet_model.compile(optimizer=Adam(learning_rate=1e-5))

# Cargar tripletas
triplets = np.load("/content/drive/MyDrive/triplets.npy")
triplets = triplets.astype(np.float32) / 255.0  # Normalizar
print("Shape triplets:", triplets.shape)  # (N, 3, 160, 160, 3)


# Entrenar el modelo con las tripletas
history = triplet_model.fit(
    [triplets[:, 0], triplets[:, 1], triplets[:, 2]],
    epochs=3,  # ✅ Ajustado para que sea menos intrusivo
    batch_size=32,
    validation_split=0.1
)

# Guardar los pesos del modelo entrenado
facenet.save_weights('/content/drive/MyDrive/facetune_E3C10.weights.h5')
