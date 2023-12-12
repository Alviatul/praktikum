import os
import time
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from pytorch_tabnet.tab_model import TabNetClassifier

# Load the TabNet model from h5 file using load_model
model_path = './model.zip'  # Ganti dengan nama file model tanpa ekstensi .h5
class_list = {
    'Income < 50k': 0,
    'Income > 50k': 1
}

# Inisialisasi Aplikasi
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('kode.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Coba untuk memuat model
            loaded_model = TabNetClassifier()
            loaded_model.load_model(model_path)
        except FileNotFoundError:
            # Tangani kesalahan jika file tidak ditemukan
            return render_template('error.html', message="Model file not found.")

        # Dapatkan data input dari form
        age = float(request.form['age'])
        edu = float(request.form['education'])
        ocu = float(request.form['occupation'])
        hours = float(request.form['hours_per_week'])
        country = float(request.form['native_country'])
        start = time.time()
        # Lakukan prediksi menggunakan model yang dimuat
        probabilities = loaded_model.predict(np.array([[age, edu, ocu, hours, country]]))
        runtimes = round(time.time() - start, 4)
        # Konversi list ke string sebelum mengembalikannya
        result = probabilities.tolist()[0]
        prediction_label = list(class_list.keys())[result]

        # Tampilkan prediksi di halaman hasil
        return render_template('/prediksi.html', prediction=prediction_label, runtime=runtimes)
# Jalankan Aplikasi
if __name__ == '__main__':
    # Jalankan aplikasinya
    app.run(debug=True)
