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
<div style="font-size: 13px; line-height: 1.6; margin-left: 10px;">
[1] Dessler, G. (2020). *Human Resource Management* (16th ed.). Pearson Education.  
[2] Cascio, W. F. (2015). *Managing Human Resources: Productivity, Quality of Work Life, Profits* (10th ed.). McGraw-Hill Education.  
[3] Nguyen, T., & Duong, M. (2021). “Predicting Employee Turnover Using Machine Learning Techniques.” *Journal of Human Resource Analytics,* 5(3), 45–57.  
[4] IBM HR Analytics Employee Attrition & Performance Dataset (2019). *Kaggle.*  
[https://www.kaggle.com/datasets/pavansubhashh/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/datasets/pavansubhashh/ibm-hr-analytics-attrition-dataset)
</div>

---

