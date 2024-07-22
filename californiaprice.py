#kütüphaneler
import pandas as pd 
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
#veriyi yükle
data = pd.read_csv("housing.csv")
data.head()

#kolon isimlerini inceleme
data.columns = ["boylam","enlem","ortalama_ev_yaşı","toplam_oda","toplam_yatak_odası","nüfus","hane_sayısı","ortalama_gelir","ev_fiyatı","okyanusa_yakınlık"]

#verinin ilk 10 satırı
data.head(10)

#nan değer kontrolü
data.isna().sum()
#verinin miktarı
data.shape
# na verinin ortalama ile dolduurlması
data["toplam_yatak_odası"].fillna((data["toplam_yatak_odası"].mean()),inplace=True)
#tekrar na kontrol
print(data.isna().sum())
#tekrar veri büyüklüğü kontrol
print(data.shape)


# Fiyat Dağılımı
sbn.histplot(data['ev_fiyatı'], bins=50, kde=True)
plt.title('Ev Fiyatı Dağılımı')
plt.xlabel('Ev Fiyatı')
plt.ylabel('Sıklık')
plt.show()

# Ortalama Gelir ile Ev Fiyatı Arasındaki İlişki
sbn.scatterplot(x='ortalama_gelir', y='ev_fiyatı', data=data)
plt.title('Ortalama Gelir ile Ev Fiyatı İlişkisi')
plt.xlabel('Ortalama Gelir')
plt.ylabel('Ev Fiyatı')
plt.show()

# Ortalama Ev Yaşı ile Ev Fiyatı Arasındaki İlişki
sbn.scatterplot(x='ortalama_ev_yaşı', y='ev_fiyatı', data=data)
plt.title('Ortalama Ev Yaşı ile Ev Fiyatı İlişkisi')
plt.xlabel('Ortalama Ev Yaşı')
plt.ylabel('Ev Fiyatı')
plt.show()

#label encoder yüklenmesi
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["okyanusa_yakınlık"] = le.fit_transform(data["okyanusa_yakınlık"])
#encoderdan sonra ilk 5 satır kontrol
print(data.head())

#heat map
cor = data.corr()
sbn.heatmap(cor,annot = True)

#verileri x y diye ayır ve kontrol et
x = data.drop(["ev_fiyatı"],axis = 1)
y= data.iloc[:,-2]

y.head()
#veriyi describe et
data.info()

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
#train shape test
y_train.shape

# test shape test
x_test.shape

#fit ve Transform et
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
x_train

# Gerekli kütüphaneleri yükle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Linear Regression ile modeli eğit ve tahmin al
lr = LinearRegression()
lr.fit(x_train, y_train)
prediction_lr = lr.predict(x_test)
r2_lr = r2_score(y_test, prediction_lr)
print('Linear Regression R2 Score:', r2_lr)

# Random Forest ile modeli eğit ve tahmin al
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
prediction_rfr = rfr.predict(x_test)
r2_rfr = r2_score(y_test, prediction_rfr)
print('Random Forest R2 Score:', r2_rfr)

# Gradient Boosting ile modeli eğit ve tahmin al
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
prediction_gbr = gbr.predict(x_test)
r2_gbr = r2_score(y_test, prediction_gbr)
print('Gradient Boosting R2 Score:', r2_gbr)

# XGBoost ile modeli eğit ve tahmin al
xgbr = xgb.XGBRegressor()
xgbr.fit(x_train, y_train)
prediction_xgbr = xgbr.predict(x_test)
r2_xgbr = r2_score(y_test, prediction_xgbr)
print('XGBoost R2 Score:', r2_xgbr)

# Modellerin R2 skorlarını karşılaştırma
model_scores = {
    'Linear Regression': r2_lr,
    'Random Forest': r2_rfr,
    'Gradient Boosting': r2_gbr,
    'XGBoost': r2_xgbr
}

# En iyi modeli bul ve çıktı al
best_model = max(model_scores, key=model_scores.get)
print(f"En iyi model: {best_model} (R2 Score: {model_scores[best_model]:.4f})")
