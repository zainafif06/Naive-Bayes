{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Prediksi Kanker Payudara dengan Naive Bayes\n",
    "\n",
    "Memprediksi apakah tumor bersifat ganas (malignant) atau jinak (benign) menggunakan algoritma Naive Bayes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Library yang Dibutuhkan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.metrics import confusion_matrix, classification_report, accuracy_score\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset kanker payudara dari sklearn\n",
    "data = load_breast_cancer()\n",
    "\n",
    "# Konversi ke dataframe pandas\n",
    "df = pd.DataFrame(data.data, columns=data.feature_names)\n",
    "df['target'] = data.target\n",
    "\n",
    "# Mapping target value ke label\n",
    "df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})\n",
    "\n",
    "# Tampilkan 5 data pertama\n",
    "print(\"Data Pertama:\")\n",
    "display(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Preprocessing Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pisahkan fitur dan target\n",
    "X = df.drop(['target', 'diagnosis'], axis=1)\n",
    "y = df['target']\n",
    "\n",
    "# Bagi data menjadi training dan testing set\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "\n",
    "# Normalisasi data\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Membangun Model Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Inisialisasi model Naive Bayes\n",
    "model = GaussianNB()\n",
    "\n",
    "# Training model\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Prediksi\n",
    "y_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluasi Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Hitung akurasi\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Akurasi Model: {accuracy:.2f}\")\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "print(\"\\nConfusion Matrix:\")\n",
    "print(cm)\n",
    "\n",
    "# Classification Report\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Visualisasi Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualisasi Confusion Matrix\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
    "            xticklabels=['Malignant', 'Benign'], \n",
    "            yticklabels=['Malignant', 'Benign'])\n",
    "plt.title('Confusion Matrix')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Interpretasi Hasil\n",
    "\n",
    "- **Confusion Matrix**: Menunjukkan jumlah prediksi benar dan salah\n",
    "  - True Negative (TN): Prediksi benar Malignant\n",
    "  - False Positive (FP): Prediksi salah Benign (seharusnya Malignant)\n",
    "  - False Negative (FN): Prediksi salah Malignant (seharusnya Benign)\n",
    "  - True Positive (TP): Prediksi benar Benign\n",
    "\n",
    "- **Classification Report**: Menunjukkan metrik evaluasi seperti precision, recall, dan f1-score\n",
    "\n",
    "- **Akurasi**: Persentase prediksi yang benar secara keseluruhan"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}