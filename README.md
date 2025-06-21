# Fine-Tuning de FaceNet con Tripletas

Repositorio enfocado en la generación de tripletas y el entrenamiento del modelo FaceNet, adaptado para el "sistema de control de acceso a áreas restringidas mediante reconocimiento facial y generación de códigos OTP".

## 📄 Descripción

Este módulo corresponde a la etapa de entrenamiento del sistema biométrico, donde se ajusta un modelo de reconocimiento facial mediante el método de **aprendizaje por tripletas (triplet loss)**. 
Las tripletas (ancla, positiva, negativa) se generan a partir del dataset preprocesado y se utilizan para afinar el modelo FaceNet, mejorando su capacidad de distinguir identidades con tecnicas de acondicionamiento.

Este modelo es posteriormente utilizado por el microservicio `extraccionEmbedding` del sistema desplegado en Google Cloud.

## 🧠 Tecnologías utilizadas

- Python 3.x
- TensorFlow / Keras
- NumPy
- FaceNet (modelo base)
