import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.pipeline import Pipeline
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle





test_data = pd.read_csv("carInsurance_test.csv")
train_data = pd.read_csv("carInsurance_train.csv")

print("\n The Detail understanding of Features and Number of Samples within Dataset")
print("\n The training data has {0} samples and {1} features.".format(train_data.shape[0], train_data.shape[1]-1))
print("\n The testing data has {0} samples and {1} features.".format(test_data.shape[0], test_data.shape[1]-1))

print("\n Misssing Values per Column(Feature) IN Training DataSet")
print(train_data.isnull().sum())

print("\n Misssing Values per Column(Feature) IN Testing DataSet")
print(test_data.isnull().sum())

#Find out the Outliers
def hist_matrix(data):
    numeric_cols = [col for col in data if data[col].dtype!= "O"]
    fig, ax = plt.subplots(nrows = 4, ncols = 3,figsize = (16,10))
    fig.subplots_adjust(hspace = 0.5)
    x=0
    y=0
    for i in numeric_cols:
        ax[y,x].hist(data[i])
        ax[y,x].set_title("{}".format(i))
        x+=1
        if x == 3:
            x-=3
            y+=1
    return

hist_matrix(train_data)

def handleMissingValues(data):
    #Handle Missing Values By filling the categorical data with the modal value 
    #(when there are not a significant number of missings)

    #Education
    data['Education'] = data['Education'].fillna(data['Education'].mode()[0])
    #Job
    data['Job'] = data['Job'].fillna(data['Job'].mode()[0])
    #Communication
    data['Communication'] = data['Communication'].fillna('Missing')
    #Outcome
    data['Outcome'] = data['Outcome'].fillna('Mising')
    return data


train_data_without_missing = handleMissingValues(train_data)

print("\n Train DataSet Without Missing Values")
print(train_data_without_missing.isnull().sum())

test_data_without_missing = handleMissingValues(test_data)
print("\n Test DataSet Without Missing Values")
print(test_data_without_missing.isnull().sum())

#Calculate CAll Duration And Remove Call Start and End 
def calculateCallDuration(data):
    for i in ['CallStart', 'CallEnd']:
        data[i] = pd.to_datetime(data[i])
    data['CallDur'] = ((data['CallEnd']-data['CallStart']).dt.seconds)/60
    return data

train_data_with_call_duration = calculateCallDuration(train_data_without_missing)

print("\n Train DataSet With CAll Duration")
print(train_data_with_call_duration)

test_data_with_call_duration = calculateCallDuration(test_data_without_missing)
print("\n Test DataSet Without Missing Values")
print(test_data_with_call_duration)


# Handling the OUtliers
sns.boxplot(x='Balance',data=train_data_with_call_duration,palette='hls');
# Maximum value in Balance field
train_data_with_call_duration.Balance.max()
# Looking at the particular maximum value in the dataframe
train_data_with_call_duration[train_data_with_call_duration['Balance'] == 98417]
# Dropping the index value corresponding to the outlier
train_data_with_balance_outlier_removed = train_data_with_call_duration.drop(train_data_with_call_duration.index[1742]);
sns.boxplot(x='Balance',data=train_data_with_balance_outlier_removed,palette='hls');

sns.boxplot(x='PrevAttempts',data=train_data_with_balance_outlier_removed,palette='hls');
train_data_with_balance_outlier_removed.PrevAttempts.max()
# Looking at the particular maximum value in the dataframe
train_data_with_balance_outlier_removed[train_data_with_balance_outlier_removed['PrevAttempts'] == 58]
# Dropping the index value corresponding to the outlier
train_data_with_prev_attempt_outlier_removed = train_data_with_call_duration.drop(train_data_with_call_duration.index[2354]);
sns.boxplot(x='PrevAttempts',data=train_data_with_prev_attempt_outlier_removed,palette='hls');

print(train_data_with_prev_attempt_outlier_removed)

# Remove the CallStart & Call End and 
def handleCallData(data):

 data['CallHour'] = data['CallStart'].dt.hour
 data = data.drop(['CallStart', 'CallEnd'], axis = 1)
 return data

finalDataFrame = handleCallData(train_data_with_prev_attempt_outlier_removed)

print(finalDataFrame)

#Setting up correlation for our dataframe and passing it to seaborn heatmap function
sns.set(style="white")
corr = finalDataFrame.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});




#Split the training data into the targets and features
y = finalDataFrame['CarInsurance'].copy()
x_cols = [col for col in finalDataFrame.columns if col != "CarInsurance"]
x = finalDataFrame[x_cols].copy()


#scale the numeric variables using a MinMaxScaler 
numeric_cols = [col for col in x if x[col].dtype != "O"]
numeric_transformer = Pipeline(steps = [(
                        'scaler', MinMaxScaler())])

#one-hot encode the categorical variables
categorical_cols = [col for col in x if col not in numeric_cols]
categorical_transformer = Pipeline(steps=[(
                            'ohe', OneHotEncoder(drop = 'first'))])
preprocessor = ColumnTransformer(transformers = [
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)])

#Model Implementation
randomForest = Pipeline(steps = [
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators = 100, random_state=0))])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
randomForest.fit(x_train, y_train)
preds = randomForest.predict(x_test)
x_test.to_csv(r'/Users/nilesh/Desktop/j.csv', index = False, header=True)
print("Model Accuracy: {}".format(round(accuracy_score(y_test, preds),4)*100))
print(classification_report(y_test, preds))

    
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt = ".0f")
plt.yticks([1.5,0.5], ['Did not Buy', 'Did Buy'])
plt.xticks([1.5,0.5], ['Did Buy', 'Did not Buy'])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")



names = {"feature" : numeric_cols + list(randomForest['preprocessor'].transformers_[1][1]['ohe'].get_feature_names(categorical_cols))}
imp = {'importances' : list(randomForest.steps[1][1].feature_importances_)}
feature_importances = {**names, **imp}
feature_importances_df = pd.DataFrame(feature_importances) 

feature_importances_df = feature_importances_df.sort_values(by = 'importances', ascending = True)
feature_importances_df

plt.figure(figsize = (16,10))
plt.title("Feature Importances from Random Forest Classifier Model")
plt.barh(feature_importances_df['feature'], feature_importances_df['importances'])

rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, randomForest.predict_proba(x_test)[:,1])
plt.plot(rfc_fpr, rfc_tpr, label='Random Forest')
plt.plot([0,1],[0,1],label='Base Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


pickle.dump(randomForest, open('final_model.sav', 'wb'))
