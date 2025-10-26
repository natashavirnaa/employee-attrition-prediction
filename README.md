# Laporan Proyek Machine Learning 

## Employee Attrition Prediction

**Natasha Virna Arthilita** — *5003231020*  
**Violita Inggar Ariana** — *5003231021*  
**Fahmadina Sophia** — *5003231122*

## 🗂️ Daftar Isi

1. [Domain Proyek: Manajemen Sumber Daya](#1-domain-proyek-manajamen-sumber-daya)  
   - [Referensi](#referensi)  
2. [Business Understanding](#2-business-understanding)  
   - [Problem Statements](#problem-statements)  
   - [Goals](#goals)  
   - [Solution Statements](#solution-statements)  
   - [Project Benefits](#project-benefits)  
3. [Data Understanding](#3-data-understanding)  
   - [Sumber Data](#sumber-data)  
   - [Deskripsi Fitur](#deskripsi-fitur)  
   - [Penjelasan Kontekstual Fitur](#penjelasan-kontekstual-fitur)  
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
     - [Missing Value & Outliers](#missing-value--outliers)  
     - [Univariate Analysis](#univariate-analysis)  
     - [Multivariate Analysis](#multivariate-analysis)  
     - [Kesimpulan EDA](#kesimpulan-eda)  
4. [Data Preparation](#4-data-preparation)  
   - [Label Encoding dengan Mapping pada Fitur Target](#label-encoding-dengan-mapping-pada-fitur-target)  
   - [Splitting Dataset](#splitting-dataset)  
   - [Feature Engineering, Data Cleaning and Preprocessing](#feature-engineering-data-cleaning-and-preprocessing)  
5. [Model Training, Comparison, Selection and Tuning](#5-model-training-comparison-selection-and-tuning)  
   - [Model Selection](#model-selection)  
   - [Feature Selection](#feature-selection)  
   - [Hyperparameter Tuning](#hyperparameter-tuning)  
6. [Model Testing and Evaluation](#6-model-testing-and-evaluation)  
   - [Data Test Predict](#data-test-predict)  
   - [Best Model Evaluation](#best-model-evaluation)  
     - [Classification Report](#classification-report)  
     - [Metode Evaluasi Lanjutan](#metode-evaluasi-lanjutan)  
     - [Confusion Matrix](#confusion-matrix)  
     - [Plot ROC-AUC Curve](#plot-roc-auc-curve)  
     - [Plot PR-AUC Curve](#plot-pr-auc-curve)  
7. [Save Best Model](#7-save-best-model)  
8. [Model Interpretation](#8-model-interpretation)  
   - [Interpretation with SHAP Values](#interpretation-with-shap-values)  
   - [Feature Importance](#feature-importance)  
9. [Financial Result](#9-financial-result)  
10. [Conclusions](#10-conclusions)  
    - [Ringkasan Proyek](#ringkasan-proyek)  
    - [Hasil dan Evaluasi Model](#hasil-dan-evaluasi-model)  
    - [Penanganan Ketidakseimbangan Data](#penanganan-ketidakseimbangan-data)  
    - [Interpretasi dan Validasi Model](#interpretasi-dan-validasi-model)  
    - [Estimasi Nilai Finansial](#estimasi-nilai-finansial)  
    - [Langkah Selanjutnya](#langkah-selanjutnya)

## Domain Proyek : Manajemen Sumber Daya
![Employee Attrition](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/employe-attrition.png?raw=true)
Di dunia bisnis modern, pengelolaan sumber daya manusia menjadi faktor kunci bagi keberlanjutan dan daya saing perusahaan [1]. Salah satu tantangan utama dalam bidang ini adalah tingginya tingkat attrition atau employee turnover, yaitu kondisi ketika karyawan meninggalkan perusahaan baik secara sukarela maupun tidak. Tingkat turnover yang tinggi dapat menimbulkan biaya besar bagi perusahaan, mengganggu produktivitas, serta berdampak negatif terhadap moral tim [2].
Untuk mengatasinya, banyak organisasi kini mengandalkan pendekatan berbasis data guna memahami faktor-faktor yang memengaruhi keputusan karyawan dalam bertahan atau keluar dari perusahaan [3]. Dengan memanfaatkan analisis prediktif, perusahaan dapat mengidentifikasi pola perilaku karyawan, mendeteksi risiko turnover lebih dini, serta merancang strategi retensi yang lebih efektif.
Dataset yang digunakan dalam proyek ini mencakup berbagai aspek profil karyawan, lingkungan kerja, dan status karyawan (bertahan atau keluar). Variabel-variabel tersebut meliputi faktor demografis, jabatan, jam lembur, tingkat kepuasan kerja, dan beban kerja, yang dapat digunakan untuk membangun model prediktif employee attrition dan mengeksplorasi fitur-fitur yang paling berpengaruh terhadap keputusan karyawan [4].

---

**Referensi**
<div style="font-size: 10px; line-height: 1.6; margin-left: 10px;">
[1] Dessler, G. (2020). <i>Human Resource Management</i> (16th ed.). Pearson Education.<br>
[2] Cascio, W. F. (2015). <i>Managing Human Resources: Productivity, Quality of Work Life, Profits</i> (10th ed.). McGraw-Hill Education.<br>
[3] Nguyen, T., & Duong, M. (2021). “Predicting Employee Turnover Using Machine Learning Techniques.” <i>Journal of Human Resource Analytics,</i> 5(3), 45–57.<br>
[4] IBM HR Analytics Employee Attrition & Performance Dataset (2019). <i>Kaggle.</i><br>
<a href="https://www.kaggle.com/datasets/pavansubhashh/ibm-hr-analytics-attrition-dataset">https://www.kaggle.com/datasets/pavansubhashh/ibm-hr-analytics-attrition-dataset</a>
</div>

---

## Business Understanding
### Problem Statemnents
Dalam dunia kerja modern, setiap perusahaan berusaha mempertahankan karyawan berprestasi untuk menjaga stabilitas operasional dan daya saing. Namun, tingginya tingkat employee attrition atau turnover menjadi tantangan serius bagi manajemen sumber daya manusia. Fenomena ini dapat menyebabkan peningkatan biaya rekrutmen dan pelatihan, penurunan produktivitas, serta menurunnya moral kerja tim secara keseluruhan.

Berdasarkan hal tersebut, pernyataan masalah yang diangkat dalam proyek ini adalah sebagai berikut: 
1. **Pernyataan Masalah 1:** Bagaimana mengidentifikasi faktor-faktor utama yang memengaruhi keputusan karyawan untuk keluar dari perusahaan?  
2. **Pernyataan Masalah 2:** Bagaimana membangun model prediktif yang mampu memperkirakan kemungkinan seorang karyawan akan keluar dengan akurasi yang tinggi?  
3. **Pernyataan Masalah 3:** Bagaimana memanfaatkan hasil analisis tersebut untuk merumuskan strategi retensi karyawan yang efektif dan berbasis data?

### Goals
Untuk menjawab pernyataan masalah di atas, tujuan proyek ini adalah:  
1. **Tujuan 1:** Melakukan eksplorasi dan analisis terhadap data karyawan untuk menemukan pola dan variabel yang berkorelasi dengan perilaku attrition.
2. **Tujuan 2:** Membangun model prediktif berbasis machine learning untuk menghitung probabilitas seorang karyawan akan keluar dari perusahaan.
3. **Tujuan 3:** Menyediakan insight dan rekomendasi strategis yang dapat membantu manajemen dalam meningkatkan retensi karyawan dan menciptakan lingkungan kerja yang lebih produktif.

### Solution Statements
Untuk mencapai tujuan tersebut, solusi yang akan diterapkan meliputi:  
- **Eksperimen dengan Berbagai Algoritma Klasifikasi**
  Membangun dan membandingkan performa beberapa algoritma seperti:
  - Logistic Regression
  - Random Forest
  - XGBoost / LightGBM
- **Optimasi Model dengan Hyperparameter Tuning**
  Menggunakan pendekatan seperti Grid Search atau Optuna (Bayesian Optimization) untuk menemukan konfigurasi parameter terbaik.
- **Evaluasi Model dengan Metrik yang Relevan**
  Menerapkan metrik evaluasi seperti:
  - Accuracy — untuk mengukur kinerja keseluruhan model.
  - Precision, Recall, dan F1-Score — untuk menilai performa pada kelas minoritas (attrition).
  - ROC-AUC — untuk mengevaluasi kemampuan model dalam membedakan karyawan yang bertahan dan keluar.
  - Confusion Matrix — untuk memvisualisasikan hasil prediksi model.
- **Analisis Fitur Dan visualisasi**
  Menyajikan visualisasi seperti feature importance dan correlation heatmap untuk menginterpretasikan fitur-fitur utama yang berkontribusi pada analisis.

### Project Benefit
Dengan implementasi solusi ini, manfaat yang diharapkan antara lain:
- **Efisiensi Biaya SDM:** Mengurangi biaya rekrutmen dan pelatihan akibat tingginya turnover.
- **Peningkatan Retensi Karyawan:** Mengidentifikasi dan menindaklanjuti karyawan dengan risiko tinggi untuk keluar.
- **Pengambilan Keputusan Berbasis Data:** Membantu manajemen dalam menyusun strategi retensi dan kebijakan HR yang lebih tepat sasaran.
- **Perbaikan Lingkungan Kerja:** Memberikan insight terhadap faktor-faktor yang menyebabkan ketidakpuasan dan menurunkan motivasi kerja.
- **Produktivitas yang Lebih Stabil:** Menjaga kontinuitas tim dan mengurangi dampak negatif dari pergantian karyawan.





