# UAS_KecerdasanBuatan
# Student Depression Prediction using Machine Learning

Analisis prediktif untuk mengidentifikasi faktor risiko depresi pada mahasiswa menggunakan algoritma **Logistic Regression** dan **Random Forest Classifier** dengan metodologi **CRISP-DM**. Penelitian ini bertujuan membangun model klasifikasi yang mampu mendeteksi mahasiswa yang berisiko mengalami depresi sehingga dapat dimanfaatkan sebagai dasar pengembangan **Early Warning System (EWS)** di lingkungan perguruan tinggi. :contentReference[oaicite:0]{index=0}

---

## Latar Belakang

Kesehatan mental mahasiswa merupakan salah satu isu penting yang memengaruhi prestasi akademik, kualitas hidup, hingga risiko putus kuliah. Dengan memanfaatkan teknik Machine Learning, penelitian ini membangun model prediksi depresi berdasarkan berbagai faktor akademik, finansial, dan psikososial. :contentReference[oaicite:1]{index=1}

---

## Tujuan Penelitian

- Membangun pipeline preprocessing data yang baik.
- Membandingkan performa Logistic Regression dan Random Forest.
- Mengidentifikasi faktor-faktor utama yang memengaruhi depresi mahasiswa.
- Memberikan dasar pengembangan sistem deteksi dini kesehatan mental mahasiswa. :contentReference[oaicite:2]{index=2}

---

## Dataset

Dataset yang digunakan adalah:

**Student Depression Dataset**

Dataset berisi informasi mengenai:

- Umur
- Jenis Kelamin
- CGPA
- Academic Pressure
- Financial Stress
- Study Hours
- Sleep Duration
- Dietary Habits
- Family History
- Suicidal Thoughts
- Depression (Target)

---

## Metodologi

Penelitian menggunakan metodologi **CRISP-DM (Cross Industry Standard Process for Data Mining)** yang terdiri dari:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation

:contentReference[oaicite:3]{index=3}

---

## Data Preprocessing

Tahapan preprocessing meliputi:

- Membersihkan nama kolom
- Menghapus missing value
- Menghapus data duplikat
- Feature Selection
- Label Encoding
- Train-Test Split (80:20)
- Feature Scaling menggunakan StandardScaler

Pipeline preprocessing:

```
Raw Dataset
      │
      ▼
Cleaning
      │
      ▼
Missing Value Handling
      │
      ▼
Duplicate Removal
      │
      ▼
Feature Selection
      │
      ▼
Label Encoding
      │
      ▼
Train-Test Split
      │
      ▼
StandardScaler
```

:contentReference[oaicite:4]{index=4}

---

## Algoritma Machine Learning

Dua algoritma dibandingkan:

### Logistic Regression

Model klasifikasi linier yang digunakan sebagai baseline.

### Random Forest Classifier

Model ensemble berbasis Decision Tree yang menggunakan teknik Bootstrap Aggregating (Bagging).

:contentReference[oaicite:5]{index=5}

---

## Evaluasi Model

Evaluasi dilakukan menggunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

:contentReference[oaicite:6]{index=6}

---

## Hasil Penelitian

Hasil penelitian menunjukkan bahwa:

- Random Forest memiliki performa lebih baik dibandingkan Logistic Regression.
- Academic Pressure merupakan faktor yang sangat berpengaruh terhadap depresi.
- Financial Stress memiliki hubungan kuat dengan depresi mahasiswa.
- Mahasiswa yang memiliki suicidal thoughts hampir seluruhnya termasuk kelompok depresi.
- CGPA cenderung menurun pada mahasiswa yang mengalami depresi.

:contentReference[oaicite:7]{index=7}

---

## Faktor Risiko Utama

Penelitian menemukan beberapa faktor utama penyebab depresi:

- Academic Pressure tinggi
- Financial Stress tinggi
- Jam belajar berlebihan
- Ide bunuh diri (Suicidal Thoughts)
- Penurunan CGPA

:contentReference[oaicite:8]{index=8}

---

## Library yang Digunakan

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Struktur Project

```
Student-Depression-Prediction/
│
├── dataset/
│   └── Student Depression Dataset.csv
│
├── notebook/
│   └── AI_TB_UAS_Student_Depression_Dataset.ipynb
│
├── images/
│
├── README.md
│
└── requirements.txt
```

---

## Cara Menjalankan

### Clone Repository

```bash
git clone https://github.com/username/student-depression-prediction.git
```

### Install Dependency

```bash
pip install -r requirements.txt
```

### Jalankan Notebook

```bash
jupyter notebook
```

Kemudian buka file:

```
AI_TB_UAS_Student_Depression_Dataset.ipynb
```

---

## Referensi

- Breiman, L. (2001). Random Forests.
- Pedregosa et al. (2011). Scikit-Learn.
- Hosmer & Lemeshow (2013). Applied Logistic Regression.
- Wirth & Hipp (2000). CRISP-DM.
- American Psychiatric Association (2013). DSM-5.

:contentReference[oaicite:9]{index=9}

---

## Tim

**Kelompok 2**

- Sri Sukma Fitria (2406037)
- Neng Isan Lopiani (2406179)

Program Studi Teknik Informatika  
Institut Teknologi Garut

:contentReference[oaicite:10]{index=10}

---

## Lisensi

Project ini dibuat untuk keperluan akademik sebagai Tugas Akhir Mata Kuliah Kecerdasan Buatan (Artificial Intelligence).
