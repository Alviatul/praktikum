# Proyek Deploy Model menggunakan Flask

Proyek ini mendemonstrasikan cara mendeploy model machine learning menggunakan Flask. Model ini memprediksi tingkat pendapatan berdasarkan dataset Pendapatan Dewasa.

## Penggunaan

### 1. Clone repositori GitHub

```bash
git clone https://github.com/Alviatul/praktikum.git
```

### 2. Pindah ke direktori proyek

```bash
cd praktikum
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi Flask

```bash
python app.py
```

Kunjungi [http://localhost:5000](http://localhost:5000) di peramban web Anda untuk berinteraksi dengan model yang telah dideploy.

## Dataset

Model ini dilatih menggunakan dataset Pendapatan Dewasa, yang berisi informasi tentang individu dan tingkat pendapatannya.

### Sumber Dataset

[UCI Machine Learning Repository - Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

## Struktur Proyek

- `app.py`: Berkas utama aplikasi Flask.
- `templates/`: Berisi template HTML untuk merender halaman web.
- `model/`: Berisi model machine learning yang telah dilatih.
- `requirements.txt`: Daftar semua dependensi Python untuk proyek ini.