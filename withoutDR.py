
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np

path, dirs, files = next(os.walk("positive_feature_merge"))
file_count = len(files)
data_train = []
data_test = []

for file in files:
    print(file)

    #read files to reduce dimensions
    df1 = pd.read_csv('positive_feature_merge/' + file)
    df2 = pd.read_csv('negative_feature_merge/' + file[:-7] + "neg.csv")

    df1 = df1.drop(["Unnamed: 0"], axis=1)
    df2 = df2.drop(["Unnamed: 0"], axis=1)

    df = [df1,df2]
    result = pd.concat(df, axis=0)

    df = result.drop(["Sample_Name"], axis=1)
    X = df.drop(["Label"], axis=1)
    y = pd.DataFrame(df["Label"])

    namelist=[]
for i in range(0,32):
  namelist.append("f"+str(i))
print(namelist)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Calculate and display the correlation matrix between features
correlation_matrix = X_train.corr()

# Visualize the correlation matrix as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20), dpi = 300)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

#random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = rf.predict(X_test)

#plotting confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

plot_confusion_matrix(rf, X_test, y_test)

y_score1 = rf.predict_proba(X_test)[:,1]

#plotting roc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, thresh = roc_curve(y_test, y_score1)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr)
plt.plot([0, 1.0], linestyle='--',color='gray')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')

import seaborn as sns

def plot_feature_importance(importance,names,model_type):

  #Create arrays from feature importance and feature names
  feature_importance = np.array(importance)
  print(feature_importance.shape)
  feature_names = np.array(names)
  print(feature_names.shape)


  #Create a DataFrame using a Dictionary
  data={'feature_names':feature_names,'feature_importance':feature_importance}
  fi_df = pd.DataFrame(data)

  #Sort the DataFrame in order decreasing feature importance
  fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

  #Define size of bar plot
  plt.figure(figsize=(20,20))
  #Plot Searborn bar chart
  sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
  #Add chart labels
  plt.title(model_type + 'FEATURE IMPORTANCE')
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')

plot_feature_importance(rf.feature_importances_,namelist,'RANDOM FOREST')