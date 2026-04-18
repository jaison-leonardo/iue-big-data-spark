"""
Módulo para el entrenamiento y evaluación del modelo CNN.
Consume los datos generados por src/preprocess.py
"""

import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Importar las funciones generadas previamente en preprocess
try:
    from src.preprocess import prepare_data, get_train_augmentation
except ImportError:
    from preprocess import prepare_data, get_train_augmentation

# Configurar logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_model(input_shape=(224, 224, 3), num_classes=2, use_transfer_learning=False):
    """
    Construye la arquitectura del modelo.
    Incluye una CNN tradicional y la rama de MobileNetV2 elegible por parámetro.
    """
    logging.info(f"Construyendo modelo {'MobileNetV2 (Transfer Learning)' if use_transfer_learning else 'CNN Custom'} para {num_classes} clases.")
    
    if use_transfer_learning:
        # MobileNetV2
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        base_model.trainable = False  # Congelar capas pre-entrenadas
        
        inputs = tf.keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x) # Similar al Flatten pero más eficiente para CNN base modernas
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
    else:
        # Arquitectura CNN personalizada
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5), # Capa de Dropout para evitar Overfitting
            Dense(num_classes, activation='softmax')
        ])

    # Compilación usando Adam y Categorical Crossentropy
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def plot_history(history, save_path='models/training_history.png'):
    """
    Genera y guarda un gráfico del historial de desempeño del entrenamiento.
    """
    logging.info(f"Exportando gráfica de resultados a {save_path}...")
    plt.figure(figsize=(12, 4))
    
    # Gráfica de Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Evolución de la Función de Pérdida (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfica de Precisión (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Evolución de Precisión (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(X_train, y_train, X_val, y_val, num_classes, epochs=20, batch_size=32, model_save_path='models/model.keras', use_transfer_learning=False):
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    
    model = build_model(input_shape=X_train.shape[1:], num_classes=num_classes, use_transfer_learning=use_transfer_learning)
    
    datagen = get_train_augmentation()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    logging.info("Arrancando proceso de entrenamiento con Data Augmentation en línea...")

    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=batch_size),
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        steps_per_epoch=(len(X_train) // batch_size) if len(X_train) >= batch_size else 1,
        callbacks=callbacks,
        verbose=1
    )
    plot_history(history, save_path=model_save_path.replace('.keras', '.png').replace('.h5', '.png'))
    return model

def evaluate_model(model, X_test, y_test, num_classes):
    logging.info("Evaluando desempeño absoluto en el Set de Prueba (Test)...")
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print("-" * 40)
    print("### RESULTADOS FINALES DE EVALUACIÓN ###")
    print(f"-> Test Loss: {loss:.4f}")
    print(f"-> Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("-" * 40)
    
    return loss, accuracy

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    logging.info("Invocando caché o preprocesamiento desde preprocess.py...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = prepare_data()
    except ValueError as e:
        logging.error(str(e))
        logging.error("Pausa: Faltan las imágenes en data/raw/ para entrenar. Añade las carpetas con imágenes primero.")
        return

    num_classes = len(class_names)
    logging.info(f"Clases identificadas para entrenar: {num_classes} {class_names}")
    
    if len(X_train) == 0:
         logging.error("No hay suficientes imágenes para formar lotes de entrenamiento.")
         return
         
    model_save_path = 'models/model.keras' 
    
    modelo = train_model(X_train, y_train, X_val, y_val, num_classes, 
                         epochs=15, 
                         batch_size=16, 
                         model_save_path=model_save_path,
                         use_transfer_learning=True)
                                
    evaluate_model(modelo, X_test, y_test, num_classes)
    
    with open('models/classes.json', 'w') as f:
        json.dump(class_names, f)
        
    logging.info(f"Se ha consolidado el entrenamiento. El modelo base espera inferencias en {model_save_path}")

if __name__ == "__main__":
    main()
