import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df_water = pd.read_csv('water_potability.csv')
df_water.head()
df_water_cleaned = df_water.fillna(df_water.mean())
from imblearn.over_sampling import SMOTE
x = df_water_cleaned[list(df_water_cleaned.columns)[0:-1]]
y = df_water_cleaned['Potability']
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size = 0.2, random_state = 75)
# Random Forest
model4 = RandomForestClassifier()
model4.fit(x_train, y_train)
pred = model4.predict(x_test)
accuracy4 = round(accuracy_score(y_test, pred) * 100, 2)
print(f'Accuracy of Support Vector: {accuracy4}')
print(classification_report(y_test, pred))
plt.figure(figsize = (5, 3))
sns.heatmap(confusion_matrix(y_test, pred), annot = True, fmt = '', cmap = 'viridis')
import pickle
with open("myModel.pkl","wb") as file:
    pickle.dump(model4,file) 

