


#Importing Libraries
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import joblib
plt.style.use('fivethirtyeight')
plt.figure(figsize=(10,8))
warnings.filterwarnings('ignore')

# Uploading Files from local computer
train = pd.read_csv('C:\\Users\\Asus\\Desktop\\titanic\\data\\train.csv')
test = pd.read_csv('C:\\Users\\Asus\\Desktop\\titanic\\data\\test.csv')


# function to check data information
def data_check(data):
    shape = data.shape
    naValues = data.isna().sum()
    description = data.describe()
    return shape, naValues, description


# function to extract last Name of Passengers
def get_lastName(Name):
    return Name.split(",")[0]


# Function to extract the alias of the passenger
def get_alias(Name):
    return Name.split(" ")[1].strip(".")


# Function to Convert into categorical variables the Pclass
def Pclass(Pclass):
  if  Pclass == 1:
    return 'First'
  elif Pclass == 2:
    return 'Second'
  else:
    return 'Third'


#Function for survivors
def survivors(person):
  if person == 0:
    return 'DEAD'
  else:
    return 'ALIVE'


#Function to Embar Places
def embark(place):
  if place == 'S':
    return 'Southampton'
  elif place == 'Q':
    return 'Queenstown'
  else:
    return 'Cherbourg'


# Droppinng the Cabin ,Ticket and PassengerID Columns
train.drop(columns=['Ticket','Cabin','PassengerId'],inplace=True)
test.drop(columns=['Ticket','Cabin'],inplace=True)


# Filling NaN values with average years
train.fillna(train.mean(), inplace=True)
test.fillna(train.mean(), inplace=True)

# Dropping the two rows with na values of train test
train.dropna(inplace=True)

# Create a column with the last name of the passengers
train['lastName'] = train['Name'].apply(get_lastName)

# Create a column with the Alias of the passengers
train['Alias'] = train['Name'].apply(get_alias)

# Replacing Embarked Place column values with actual names
train['Embarked'] = train.Embarked.apply(embark)

# Extraccting the Alias
alias = train['Alias'].value_counts()
alias = alias.head(4)
alias = pd.DataFrame(alias)
alias.columns = ['Alias Count']

# Plot Alias count
sns.barplot(data=alias, x=alias.index, y='Alias Count')
plt.xlabel('Passenger Alias')
plt.title('PASSENGER ALIAS COUNT')
plt.show()

# Plot embarking place count
sns.countplot(data=train, x='Embarked',palette='Accent')
plt.xlabel('Embarking Place')
plt.title('PASSENGERS EMBARKING PLACE')
plt.show()

# Plot sex count
sex_color = ['skyblue', 'pink']
sns.countplot(data=train, x='Sex',palette=sex_color)
plt.xlabel('Sex')
plt.title('PASSENGERS SEX')
plt.show()

#Plot embarking place count
sns.countplot(data=train, x='Embarked',palette='Accent',hue='Sex')
plt.xlabel('Embarking Place')
plt.title('PASSENGERS EMBARKING PLACE SPLITTED BY SEX')
plt.show()

#Plot Passenger SibSp
sns.countplot(data=train, x='SibSp',palette='gist_rainbow_r')
plt.xlabel('SibSp')
plt.title('PASSENGERS SibSp')
plt.show()

# Age Distribution
sns.distplot(train['Age'], color = 'darkred')
plt.title('DISTRIBUTION OF AGE', fontsize = 15)
plt.show()


# Fare Distribution
sns.distplot(train['Fare'], color = 'darkred')
plt.title('DISTRIBUTION OF FARE $', fontsize = 15)
plt.show()

# Plot embarking place count
sns.countplot(data=train, x='Pclass',palette='gist_rainbow_r',hue='Sex')
plt.xlabel('Ticket Class')
plt.title('TICKET CLASS DIVIDED BY SEX')
plt.show()

# Plot embarking place count
sns.countplot(data=train, x='Pclass',palette='gist_stern_r',hue='Embarked')
plt.xlabel('Ticket Class')
plt.title('TICKET CLASS FROM THE EMBARKING PLACE')
plt.show()

# Plot embarking place count
sns.countplot(data=train, x='Survived',palette='gist_stern_r')
plt.xlabel('Ticket Class')
plt.title('SURVIVORS')
plt.show()


# Plot embarking place count
sns.countplot(data=train, x='Survived',palette='bone',hue='Sex')
plt.xlabel('Survived')
plt.title('SURIVORS SPLITTED BY SEX')
plt.show()

# Plot embarking place count
sns.countplot(data=train, x='Survived',palette='Set2',hue='Embarked')
plt.xlabel('Survived')
plt.title('SURVIVORS PER EMBARKING PLACE')
plt.show()


# Plot embarking place count
sns.countplot(data=train, x='Survived',palette='Set1',hue='Pclass')
plt.xlabel('Survived')
plt.title('TICKET CLASS FROM THE EMBARKING PLACE')
plt.show()


# Plot embarking place count
sns.countplot(data=train, x='SibSp',palette='Set1',hue='Survived')
plt.xlabel('Survived')
plt.title('TICKET CLASS FROM THE EMBARKING PLACE')
plt.show()

# Create Dataframe based on the train dataset
model_df = train

# Dropping some useless columns
model_df = model_df.drop(columns=['Alias','Name','lastName'])

# create dymmy variables
model_df = pd.get_dummies(model_df)

# Create Categorical Variables for PClass
model_df['Survived'] = model_df['Survived'].apply(survivors)

plt.figure(figsize=(15,10))
sns.heatmap(model_df.corr(), annot = True, cmap = 'Wistia')
plt.style.use('fivethirtyeight')
plt.title('Heatmap for the Variables', fontsize = 15)
plt.show()

# Dropping columns
model_df = model_df.drop(columns=['Age','Fare','Embarked_Cherbourg','Embarked_Queenstown','Embarked_Southampton','SibSp','Parch'])


#MACHINE LEARNING
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier


#Data Splitt
X = model_df.iloc[:,1:6]
y = model_df.Survived.values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30)

# KNN Model
knn_classifier = KNeighborsClassifier(n_neighbors=29)
knn_classifier = knn_classifier.fit(X_train,y_train)

# Predict
prediction = knn_classifier.predict(X_test)

# model Accuracy
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
print(accuracy_score(y_test, prediction))

# Model Check with Random Values
test_knn = [[1,1,0],[1,0,1]]
print(knn_classifier.predict(test_knn))

# Save Model
joblib.dump(knn_classifier, 'knn_model_titanic.joblib')

#RANDOM FOREST CLASSIFIER
def randomForest(X_train, X_test, y_train, y_test):
  random_classifier = RandomForestClassifier(n_estimators=5, random_state=0)
  random_classifier.fit(X_train, y_train)
  random_prediction = random_classifier.predict(X_test)

  #Model Accuracy
  print(confusion_matrix(y_test, random_prediction))
  print(classification_report(y_test, random_prediction))
  print('Model Accuracy: ',accuracy_score(y_test, random_prediction))

  #Testing Model With Random Examples
  rf_test = [[3,1,0],[2,0,1]]
  print('CLASSIFICATION EXAMPLES',random_classifier.predict(rf_test))

  #Save Model
  joblib.dump(random_classifier, 'randomforest_model_titanic.joblib')
  return random_classifier


# Calling Function
randomForest(X_train, X_test, y_train, y_test)


#checking dataframe
test.head()


# Checking Data Structure from test dataset
data_check(test)

# Creating an alternative dataset to analyze
test_data = test

# Dropping Unecessaries columns
test_data = test.drop(columns=['Name', 'Age','SibSp','Parch','Fare','Embarked','PassengerId'])

# creating Dummy variables
test_data = pd.get_dummies(test_data)

# Converting Dataframe to numpy Array
df = test_data.to_numpy()

# Knearest Classifier Model to predict Survivors on test set
test_predictions = knn_classifier.predict(df)

# Creating Dataframe with resuts
survive = pd.DataFrame(test_predictions)
survive.columns = ['Survive']

# Appending survive dataframe to test
test = test.join(survive)

#Survive Predictions
sns.countplot(data=test, x='Survive',palette=sex_color)
plt.title('SURVIVORS PREDICTIONS FOR TEST DATASET')
plt.ylabel('Count')
plt.show()

#Dropping columns unecessary columns
survivors = test.drop(columns=['Pclass', 'Name','Sex','Age','SibSp','Parch','Fare','Embarked'])

# To CSV File
survivors.to_csv(r'C:\\Users\\Asus\\Desktop\\titanic\\titanic_prediction.csv')
