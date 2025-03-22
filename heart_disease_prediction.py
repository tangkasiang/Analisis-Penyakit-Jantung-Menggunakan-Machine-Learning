import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Membaca dataset
data = pd.read_csv('heart.csv')
print(data.head())
print(data.info())
print(data.describe())

# Mengatasi nilai yang hilang
print("Jumlah nilai yang hilang sebelum pengisian:")
print(data.isnull().sum())
# Mengganti pengisian nilai yang hilang dengan median
data.fillna(data.median(), inplace=True)
print("Jumlah nilai yang hilang setelah pengisian:")
print(data.isnull().sum())

# Normalisasi data
kolom_numerik = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
kolom_target = 'target'  
if kolom_target in kolom_numerik:
    kolom_numerik.remove(kolom_target)
scaler = StandardScaler()
data[kolom_numerik] = scaler.fit_transform(data[kolom_numerik])

# Analisis Eksplorasi Data (EDA)
# Visualisasi distribusi usia
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=30, kde=True)
plt.title('Distribusi Usia Pasien')
plt.xlabel('Usia')
plt.ylabel('Frekuensi')
plt.show()

# Menampilkan statistik deskriptif untuk kolom usia
print("Statistik Deskriptif Usia:")
print(data['age'].describe())

# Visualisasi hubungan antara kolesterol dan penyakit jantung
if 'cholesterol' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target', y='cholesterol', data=data)
    plt.title('Hubungan antara Kolesterol dan Penyakit Jantung')
    plt.xlabel('Penyakit Jantung (0 = Tidak, 1 = Ya)')
    plt.ylabel('Tingkat Kolesterol')
    plt.show()

# Pemilihan Fitur
X = data.drop(kolom_target, axis=1)
y = data[kolom_target]
X_new = SelectKBest(f_classif, k=5).fit_transform(X, y)
print("Fitur yang dipilih:", X.columns[SelectKBest(f_classif, k=5).fit(X, y).get_support()])

# Memisahkan data menjadi fitur dan target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menginisialisasi dan melatih model
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

model_svm = SVC(probability=True)
model_svm.fit(X_train, y_train)

# Mengganti ANN dengan KNN
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)

# Prediksi
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_svm = model_svm.predict(X_test)
y_pred_knn = model_knn.predict(X_test) 

# Metode evaluasi
def cetak_metrik(y_true, y_pred, nama_model):
    print(f"\n{nama_model}:")
    print("Akurasi:", accuracy_score(y_true, y_pred))
    print("Presisi:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Skor F1:", f1_score(y_true, y_pred))

cetak_metrik(y_test, y_pred_lr, "Regresi Logistik")
cetak_metrik(y_test, y_pred_rf, "Random Forest")
cetak_metrik(y_test, y_pred_svm, "Support Vector Machine")
cetak_metrik(y_test, y_pred_knn, "K-Nearest Neighbors")  

# Skor AUC
print("\nSkor AUC:")
print("AUC Regresi Logistik:", roc_auc_score(y_test, model_lr.predict_proba(X_test)[:, 1]))
print("AUC Random Forest:", roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1]))
print("AUC SVM:", roc_auc_score(y_test, model_svm.predict_proba(X_test)[:, 1]))
print("AUC KNN:", roc_auc_score(y_test, model_knn.predict_proba(X_test)[:, 1])) 

# Kurva ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, model_lr.predict_proba(X_test)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1])
fpr_svm, tpr_svm, _ = roc_curve(y_test, model_svm.predict_proba(X_test)[:, 1])
fpr_knn, tpr_knn, _ = roc_curve(y_test, model_knn.predict_proba(X_test)[:, 1]) 

plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, label='Regresi Logistik (area = %0.2f)' % roc_auc_score(y_test, model_lr.predict_proba(X_test)[:, 1]))
plt.plot(fpr_rf, tpr_rf, label='Random Forest (area = %0.2f)' % roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1]))
plt.plot(fpr_svm, tpr_svm, label='SVM (area = %0.2f)' % roc_auc_score(y_test, model_svm.predict_proba(X_test)[:, 1]))
plt.plot(fpr_knn, tpr_knn, label='KNN (area = %0.2f)' % roc_auc_score(y_test, model_knn.predict_proba(X_test)[:, 1]))  

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
plt.legend(loc='lower right')
plt.show()

# Menyimpan model
joblib.dump(model_lr, 'model_regresi_logistik.pkl')
joblib.dump(model_rf, 'model_random_forest.pkl')
joblib.dump(model_svm, 'model_svm.pkl')
joblib.dump(model_knn, 'model_knn.pkl')  

# Tampilan Target untuk Menunjukkan Pasien Menderita Penyakit Jantung
# Menghitung jumlah pasien berdasarkan status penyakit jantung
jumlah_target = data['target'].value_counts()

# Membuat diagram batang
plt.figure(figsize=(8, 5))
bar_plot = sns.barplot(x=jumlah_target.index, y=jumlah_target.values, hue=jumlah_target.index, palette='viridis', legend=False)
plt.title('Jumlah Pasien Berdasarkan Status Penyakit Jantung')
plt.xlabel('Status Penyakit Jantung (0 = Tidak, 1 = Ya)')
plt.ylabel('Jumlah Pasien')
plt.xticks(ticks=[0, 1], labels=['Tidak', 'Ya'])

# Menambahkan total pasien di dalam kotak pada setiap bar
for p in bar_plot.patches:
    # Mengambil tinggi dan lebar bar
    height = p.get_height()
    width = p.get_width()
    x = p.get_x() + width / 2
    y = height / 2  # Menempatkan teks di tengah bar

    # Menambahkan kotak di belakang teks
    bar_plot.annotate(f'{int(height)}', 
                      (x, y), 
                      ha='center', va='center', 
                      fontsize=8, color='white', 
                      bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

plt.show()