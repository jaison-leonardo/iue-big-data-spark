import os
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_images_and_labels(data_dir, img_size=(128, 128)):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"El directorio '{data_dir}' no existe.")

    class_names = sorted([c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))])
    
    if not class_names:
        raise ValueError(f"No se encontraron carpetas. Estructura: data_dir/clase/imagen.jpg")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    logging.info(f"Clases detectadas automáticamente: {class_to_idx}")
    
    valid_paths = []
    labels_list = []
    
    for cls_name in class_names:
        cls_dir = os.path.join(data_dir, cls_name)
        img_names = os.listdir(cls_dir)
        for img_name in img_names:
            valid_paths.append(os.path.join(cls_dir, img_name))
            labels_list.append(class_to_idx[cls_name])
            
    total_imgs = len(valid_paths)
    if total_imgs == 0:
        raise ValueError("No se obtuvieron fotos válidas para preprocesar.")
        
    # 2. Creación del Bloque Único en RAM usando Float16 para bajar memoria a menos de 2GB.
    req_gb = (total_imgs * img_size[0] * img_size[1] * 3 * 2) / (1024**3)
    logging.info(f"PRE-ASIGNANDO BLOQUE DE RAM => Resolucion {img_size[0]}x{img_size[1]} usando fp16. {req_gb:.2f} GB Est.")
    
    X = np.empty((total_imgs, img_size[0], img_size[1], 3), dtype=np.float16)
    y = np.array(labels_list, dtype=np.int32)
    
    # 3. Llenado
    current_idx = 0
    for img_path in valid_paths:
        try:
             # Carga individual directa
             img = load_img(img_path, target_size=img_size)
             img_array = img_to_array(img) / 255.0  
             X[current_idx] = img_array.astype(np.float16)
        except Exception as e:
             logging.warning(f"No se pudo cargar {img_path}: {str(e)}")
        
        current_idx += 1
        if current_idx % 2000 == 0:
            logging.info(f"--> Procesadas en memoria: {current_idx}/{total_imgs} imágenes.")

    return X, y, class_names

def split_dataset(X, y, test_size=0.15, val_size=0.15):
    logging.info("Dividiendo el dataset con proporciones especificadas...")
    val_ratio_over_temp = val_size / (1.0 - test_size)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_over_temp, random_state=42, shuffle=True, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_train_augmentation():
    logging.info("Configurando Data Augmentation para conjunto de entrenamiento...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen

def prepare_data(data_dir='data/raw', processed_dir='data/processed', img_size=(128, 128), use_cache=True): # Forzamos 128x128
    cache_path = os.path.join(processed_dir, 'dataset_preprocesado.npz')
    
    if use_cache and os.path.exists(cache_path):
        logging.info(f"Cargando dataset precesado desde caché en: {cache_path}")
        data = np.load(cache_path)
        return (data['X_train'], data['X_val'], data['X_test'], 
                data['y_train'], data['y_val'], data['y_test'], 
                data['class_names'].tolist())
    
    logging.info("Iniciando Pipeline de Preprocesamiento en Crudo (Ahorro Memoria Activado)...")
    X, y, class_names = load_images_and_labels(data_dir, img_size)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    if use_cache and os.path.exists(processed_dir):
        logging.info(f"Guardando versión compresa en {cache_path}...")
        np.savez_compressed(
            cache_path,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            class_names=class_names
        )
    
    logging.info("Pipeline completado exitosamente.")
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names
