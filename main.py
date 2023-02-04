
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor,LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.impute import KNNImputer

from warnings import filterwarnings
filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df=pd.read_csv("Hitters.csv")

"""
AtBat 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
Hits 1986-1987 sezonundaki isabet sayısı
HmRun 1986-1987 sezonundaki en değerli vuruş sayısı
Runs 1986-1987 sezonunda takımına kazandırdığı sayı
RBI Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
Walks Karşı oyuncuya yaptırılan hata sayısı
Years Oyuncunun major liginde oynama süresi (sene)
CAtBat Oyuncunun kariyeri boyunca topa vurma sayısı
CHits Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
CHmRun Oyucunun kariyeri boyunca yaptığı en değerli sayısı
CRuns Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
CRBI Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
CWalks Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
League Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
Division 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
PutOuts Oyun icinde takım arkadaşınla yardımlaşma
Assits 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
Errors 1986-1987 sezonundaki oyuncunun hata sayısı
Salary Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
NewLeague 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör
"""

#print(df.head())
"""
   AtBat  Hits  HmRun  Runs  RBI  ...  PutOuts  Assists  Errors  Salary  NewLeague
0    293    66      1    30   29  ...      446       33      20     NaN          A
1    315    81      7    24   38  ...      632       43      10   475.0          N
2    479   130     18    66   72  ...      880       82      14   480.0          A
3    496   141     20    65   78  ...      200       11       3   500.0          N
4    321    87     10    39   42  ...      805       40       4    91.5          N
"""

#print(df.info())
"""
Data columns (total 20 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   AtBat      322 non-null    int64  
 1   Hits       322 non-null    int64  
 2   HmRun      322 non-null    int64  
 3   Runs       322 non-null    int64  
 4   RBI        322 non-null    int64  
 5   Walks      322 non-null    int64  
 6   Years      322 non-null    int64  
 7   CAtBat     322 non-null    int64  
 8   CHits      322 non-null    int64  
 9   CHmRun     322 non-null    int64  
 10  CRuns      322 non-null    int64  
 11  CRBI       322 non-null    int64  
 12  CWalks     322 non-null    int64  
 13  League     322 non-null    object 
 14  Division   322 non-null    object 
 15  PutOuts    322 non-null    int64  
 16  Assists    322 non-null    int64  
 17  Errors     322 non-null    int64  
 18  Salary     263 non-null    float64
 19  NewLeague  322 non-null    object 
 """

#print(df.shape) # (322, 20)

#print(df.columns)
"""
['AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Walks', 'Years', 'CAtBat',
       'CHits', 'CHmRun', 'CRuns', 'CRBI', 'CWalks', 'League', 'Division',
       'PutOuts', 'Assists', 'Errors', 'Salary', 'NewLeague']
"""

#print(df.isnull().sum())
"""
AtBat         0
Hits          0
HmRun         0
Runs          0
RBI           0
Walks         0
Years         0
CAtBat        0
CHits         0
CHmRun        0
CRuns         0
CRBI          0
CWalks        0
League        0
Division      0
PutOuts       0
Assists       0
Errors        0
Salary       59
NewLeague     0
dtype: int64
"""

"""plt.figure()
sns.heatmap(df.corr(), annot=True, cmap="BuPu")
plt.show()

print(df.corr())"""

df["AvgCAtBat"] = df["AtBat"] / df["CAtBat"]
df["AvgCHits"] = df["Hits"] / df["CHits"]
df["AvgCHmRun"] = df["HmRun"] / df["CHmRun"]
df["AvgCruns"] = df["Runs"] / df["CRuns"]
df["AvgCRBI"] = df["RBI"] / df["CRBI"]
df["AvgCWalks"] = df["Walks"] / df["CWalks"]
#Yeni değişkenlerimizi ürettik.

#print(df.head().T)
"""
           0         1         2         3         4
AtBat      293       315       479       496       321
Hits        66        81       130       141        87
HmRun        1         7        18        20        10
Runs        30        24        66        65        39
RBI         29        38        72        78        42
Walks       14        39        76        37        30
Years        1        14         3        11         2
CAtBat     293      3449      1624      5628       396
CHits       66       835       457      1575       101
CHmRun       1        69        63       225        12
CRuns       30       321       224       828        48
CRBI        29       414       266       838        46
CWalks      14       375       263       354        33
League       A         N         A         N         N
Division     E         W         W         E         E
PutOuts    446       632       880       200       805
Assists     33        43        82        11        40
Errors      20        10        14         3         4
Salary     NaN     475.0     480.0     500.0      91.5
NewLeague    A         N         A         N         N
AvgCAtBat  1.0  0.091331  0.294951  0.088131  0.810606
AvgCHits   1.0  0.097006  0.284464  0.089524  0.861386
AvgCHmRun  1.0  0.101449  0.285714  0.088889  0.833333
AvgCruns   1.0  0.074766  0.294643  0.078502    0.8125
AvgCRBI    1.0  0.091787  0.270677  0.093079  0.913043
AvgCWalks  1.0     0.104  0.288973   0.10452  0.909091
"""

"""le = preprocessing.LabelEncoder()
le.fit()"""

"""
print(df["Years"].max()) # 24
print(df["Years"].min()) # 1
"""
df['Year_lab'] = pd.qcut(df['Years'], 6 ,labels = range(1,7))

#df = preprocessing.OneHotEncoder(drop='first').fit_transform(df)
df = pd.get_dummies(df,drop_first=True)

#print(df)

imputer = KNNImputer(n_neighbors= 4).fit_transform(df)
# en yakın 4 degerin ortalamasını alıp boş degerlere koyar
df=pd.DataFrame(imputer,columns=df.columns)
# KNNimputer ile boş değerleri doldurulmuş verimizi tekrar DataFrame formatına dönüştürdük.

# aykırı deger analizi
"""
sns.boxplot(x=df["Salary"])
plt.show()
"""

q1=df["Salary"].quantile(0.25)
q3=df["Salary"].quantile(0.75)
iqr=q3-q1
lower=q1-1.5*iqr
upper=q3+1.5*iqr
df.loc[df["Salary"]>upper,"Salary"]=upper
"""
Birinci çeyreklik (Q1) = 25% = 190.25

Üçüncü çeyreklik (Q3) = 75% = 739.375

IQR = Q3 — Q1

Üst sınır değeri = Q3 + 1.5*IQR

Alt sınır değeri = Q1 – 1.5*IQR

Üst ve Alt sınır değerleri böyle bulunur.

upper değerinin üstündeki maaş değerlerini direkt olarak upper değerine eşitledik.
"""

"""sns.boxplot(x=df["Salary"])
plt.show()
"""

# local outlier factor analizi yapmak gerekiyor
"""lof=LocalOutlierFactor(n_neighbors=20)
threshold = np.sort(lof)[7]
outlier = lof > threshold
df = df[outlier]
#-2.3670826306785706 değerinden önce gelen değerleri tamamen siliyoruz.
# Çünkü bunlar fazla anormal değerler. İstesek başka değerleri de silebilirdik
# veya hiç silmeyebilirdik de. Ama amacımız hatayı azaltmak.
"""

########### TAHMİN ################
models = []

models.append(('KNN', KNeighborsRegressor()))
models.append(('SVR', SVR()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('RandomForests', RandomForestRegressor()))
models.append(('GradientBoosting', GradientBoostingRegressor()))
models.append(('XGBoost', XGBRegressor()))
models.append(('Light GBM', LGBMRegressor()))

X = df.drop("Salary",axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

for name,model in models:
    mod = model.fit(X_train,y_train)
    y_pred = mod.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(name,rmse)
    print("-------------")

############ OUTPUT ##################
"""
KNN 281.5552424368449
-------------
SVR 384.20842787745573
-------------
CART 309.22790715492573
-------------
RandomForests 224.01298127402947
-------------
GradientBoosting 225.04958018323063
-------------
XGBoost 243.40036813515204
-------------
Light GBM 232.0935142278567
-------------
"""




