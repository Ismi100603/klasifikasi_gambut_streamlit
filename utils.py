import os
import gdown
from tensorflow.keras.models import load_model

def download_model():
    model_path = 'model/cnn_model.h5'
    if not os.path.exists(model_path):
        os.makedirs('model', exist_ok=True)
        url = 'https://drive.google.com/uc?id=1o2mKhuzwVQoGuE-pKvGyYDuDnitHG72D'
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)
