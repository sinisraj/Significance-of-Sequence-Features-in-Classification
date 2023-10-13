
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
    df2 = pd.read_csv('/negative_feature_merge/' + file[:-7] + "neg.csv")

    df1 = df1.drop(["Unnamed: 0"], axis=1)
    df2 = df2.drop(["Unnamed: 0"], axis=1)

    df = [df1,df2]
    #sdf=pso
    result = pd.concat(df, axis=0)

    df = result.drop(["Sample_Name"], axis=1)
    X = df.drop(["Label"], axis=1)
    y = pd.DataFrame(df["Label"])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # np.any(np.isnan(X_train)) #and gets False
    # np.all(np.isfinite(X_train)) #and gets True


    pca = PCA(n_components=1)
    data_pca_train = pca.fit_transform(X_train)
    explained_variance = pca.explained_variance_ratio_

    print(explained_variance)
    print("")

    data_pca_test = pca.transform(X_test)

    data_train.append(data_pca_train)
    data_test.append(data_pca_test)

data_train = pd.DataFrame(np.array(data_train).squeeze())
data_test = pd.DataFrame(np.array(data_test).squeeze())
data_test.shape

data_train = np.transpose(data_train)
data_test = np.transpose(data_test)
data_train.shape
namelist=[]
for i in range(0,32):
  namelist.append("f"+str(i))
print(namelist)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

data_train.to_csv("features_train.csv", index=False)
data_test.to_csv("features_test.csv", index=False)
y_train.to_csv("labels_train.csv", index=False)
y_test.to_csv("labels_test.csv", index=False)

#reading CSV using pandas dataframe
df11 = pd.read_csv("/content/drive/features_train.csv")

df12 = pd.read_csv("/content/drive/features_test.csv")

df13 = pd.read_csv("/content/drive/labels_train.csv")

df14 = pd.read_csv("/content/drive/labels_test.csv")

X1_train = np.array(df11)
X1_test = df12
y1_train = df13
y1_test = df14

print(X1_train.shape,y1_train.shape)
print(X1_test.shape,y1_test.shape)

#random forest
from sklearn.ensemble import RandomForestClassifier
# creating a RF classifier
rf = RandomForestClassifier(n_estimators = 100)

rf.fit(X1_train, y1_train)

#Predict the response for test dataset
y1_pred = rf.predict(X1_test)

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
print(confusion_matrix(y1_test,y1_pred))
print(classification_report(y1_test,y1_pred))

plot_confusion_matrix(rf, X1_test, y1_test)

#plot roc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, thresh = roc_curve(y1_test, y1_pred)

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
  plt.title(model_type + ' FEATURE IMPORTANCE')
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')

plot_feature_importance(rf.feature_importances_,namelist,'RANDOM FOREST')