import pandas as pd
from pandas.io.xml import preprocess_data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder #LabelEncoder usually for target data
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('cleandex.csv').set_index("No")

features = list(df.columns[2:7]) + list(df.columns[14:-1])
numerical = [features[i] for i in [1, 2, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]]
categorical = [features[i] for i in [0, 3, 4, 5, 6, 9, 10]]
target = "Mega"

X, Y = df[features], df[target]
print(X.shape)
print(Y.shape)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=1)

preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(), categorical)
    ]
)

pipe = make_pipeline(
preprocessor,
    LogisticRegression() # classifier
)

print(trainX.shape)
print(trainY.shape)
pipe.fit(trainX, trainY)