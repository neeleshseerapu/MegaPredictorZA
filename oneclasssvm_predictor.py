import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import OneClassSVM

df = pd.read_csv("cleandex.csv").set_index("No")

features = list(df.columns[2:7]) + list(df.columns[14:-1])
numerical = [features[i] for i in [1, 2, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]]
categorical = [features[i] for i in [0, 3, 4, 9, 10]]

X_mega = df[df["Mega"]==1][features]

preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical) # to turn categorical variables into numbers
    ]
)

pipe = make_pipeline(
    preprocessor,
    OneClassSVM(gamma="auto", nu=0.10)
)

pipe.fit(X_mega)

for i in range(0,len(df.index)):
    if int(pipe.predict(df.loc[[i + 1]][features])[0])==1 and df.loc[i + 1]["Mega"]==0 and pipe.score_samples(df.loc[[i + 1]][features])[0] > 1.97: #messy ah logic bro pls write comments next time
        print(df.index[i], df.loc[i+1]["Original_Name"], pipe.score_samples(df.loc[[i + 1]][features])[0])

print(df.loc[500]['Name'], pipe.score_samples(df.loc[[500]][features])[0])