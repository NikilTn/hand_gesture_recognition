
# importing the necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# converting csv to a datafaram using pandas
df = pd.read_csv(".\data\data.csv")
# drop first (time) row of dataframe as it is useless
df = df.drop(columns=df.columns[0])

# removing rows whose clas is 0
df = df.loc[df["class"] != 0]

# storing class values
df2 = df['class']

# storing remaining columns
df1 = df.drop(['class'], axis='columns')

# splitting data for testing and training
x_train, x_test, y_train, y_test = train_test_split(df1, df2, test_size=0.25)

# training the model
model = DecisionTreeClassifier(random_state=24)
model.fit(x_train, y_train)
print("training the model.....")
print("please wait")

# predicting the score
print(f"score: {model.score(x_test,y_test)}")

# if you want to store your model , you can use this commented code
# it will create a file named model_pickle and stores your model
# import pickle
# with open('model_pickle','wb') as f:
#     pickle.dump(model,f)

# you can directly load your model using thia code
# with open('model_pickle','rb') as f:
#     model=pickle.load(f)

