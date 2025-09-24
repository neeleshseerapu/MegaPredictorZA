from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 2000,  2,  3],  # 2 samples, 3 features
     [11, 2000, 13],
     [21, 22, 2000],
     [10000, 140, 5000]]
y = [1, 2, 3, 1]  # classes of each sample
clf.fit(X, y)
print(clf.predict([[7000, 1, 6000]]))