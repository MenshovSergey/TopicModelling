import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor, DummyClassifier


def get_data():
    data = pd.read_csv("p_data.csv", sep=';', parse_dates=[1, 21], dtype={"user_id_new": 'str'})
    data_posts = pd.DataFrame(pd.read_pickle("merge2"))
    data_posts = data_posts.drop(25, 1)
    result = pd.merge(data, data_posts, left_on="user_id_new", right_on=26)
    g = result.groupby("user_id_new")
    result = g.filter(lambda x: len(x) > 2)
    y = result.get("DT_Psycho")
    keys = result._info_axis.values
    for i in range(1, len(keys)):
        if type(keys[i]) == str:
            result = result.drop(keys[i], 1)
    return result.as_matrix(), y


# data, y = get_data()
#
#
# np.save("merge_data", data)
# np.save("merge_data_y", y)

# data = np.load("merge_data.npy")
# y = np.load("merge_data_y.npy")
# X_train, X_test, y_train, y_test = train_test_split(data, y, stratify=data[:, 0], test_size=0.25, random_state=42)
#
# X_train = X_train[:, 0:-1]
# X_test = X_test[:, 0:-1]


# X_train = np.ndarray.astype(X_train, float)
# X_test = np.ndarray.astype(X_test, float)


def generate_feature(X, y):
    result = []
    y_res = []
    user_ids = np.unique(X[:, 0])
    for x in sorted(user_ids):
        cur_user = []
        users = np.empty((0, X.shape[1] - 1))
        y_cur = []
        for i in range(X.shape[0]):
            if X[i][0] == x:
                users = np.vstack((users, np.ndarray.astype(X[i][1:], float)))
                y_cur.append(y[i])
        std = 0
        for k in range(X.shape[1] - 1):
            std += np.std(users[:, k])
            cur_user.append(np.mean(users[:, k]))

        cur_user.append(std)
        result.append(cur_user)

        y_res.append(y_cur[0])
    median = np.median(y_res)
    for i in range(len(y_res)):
        if y_res[i] > median:
            y_res[i] = 1
        else:
            y_res[i] = 0
    return result, y_res


path = "regression/"

X_train = np.load(path + "x_train.npy")
X_test = np.load(path + "x_test.npy")
y_train = np.load(path + "y_train.npy")
y_test = np.load(path + "y_test.npy")

# X_train, y_train = generate_feature(X_train, y_train)
# X_test, y_test = generate_feature(X_test, y_test)
#
# X_train = X_train[:, 1:]
# X_test = X_test[:, 1:]

# sc = StandardScaler()
# data_n = sc.fit_transform(data)
from sklearn.model_selection import cross_val_score

# rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=42)
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, random_state=42)
# rf = GradientBoostingClassifier(random_state=42 )
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print"R^2 = " + str((rf.score(X_test, y_test)))

dfr = DummyRegressor()
# dfr = DummyClassifier()

dfr.fit(X_train, y_train)
y_pred = dfr.predict(X_test)
print(dfr.score(X_test, y_test))
