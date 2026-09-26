# BigMart Sales Prediction
Model machine learning untuk memprediksi Item_Outlet_Sales BigMart berdasarkan karakteristik produk dan outlet, sebagai dasar strategi stok, harga, dan penempatan produk.

# Problem Statement
BigMart punya data penjualan tapi belum punya model prediksi sales yang akurat, dan belum tahu variabel apa yang paling berpengaruh terhadap penjualan tiap outlet.

# Dataset
Train.csv — 8.523 baris, 12 kolom
Fitur produk: Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP
Fitur outlet: Outlet_Size, Outlet_Location_Type, Outlet_Type, Outlet_Establishment_Year
Target: Item_Outlet_Sales

# Alur Proyek
Data Cleaning — imputasi Item_Weight (mean per item), Outlet_Size (modus per outlet type), perbaikan kategori Item_Fat_Content, koreksi Item_Visibility = 0
EDA — analisis distribusi, univariate, bivariate, dan korelasi
Feature Engineering — binary/label/one-hot encoding, scaling (Robust/MinMax/Standard)
Modeling — Linear Regression, Ridge, Lasso, Decision Tree, Random Forest
Hyperparameter Tuning — GridSearchCV pada Random Forest

# Hasil Model
Model	RMSE	R²
Random Forest (tuned)	1033.95	0.607
Random Forest (baseline)	1062.56	0.585
Ridge	1144.45	0.518
Lasso	1144.48	0.518
Linear Regression	1144.48	0.518
Decision Tree	1496.22	0.176

Best model: Random Forest Regressor (tuned) max_depth=20, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=200

# Key Insight
Item_MRP adalah faktor paling dominan terhadap sales, diikuti Outlet_Type.
Karakteristik outlet (tipe, ukuran, lokasi) lebih menentukan sales daripada atribut produk individual.
Hubungan antar variabel bersifat non-linear → model tree-based/ensemble mengungguli model linear.

#Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
