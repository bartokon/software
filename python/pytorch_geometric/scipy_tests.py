from os.path import exists
import numpy as np
from dataset_torchstudio import ModelNet40_n3
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
def get_data(dataset, name = ""):
    if (exists(f"{name}_x.npy") and exists(f"{name}_y.npy")):
        return np.load(f"{name}_x.npy"), np.load(f"{name}_y.npy")
    x_train = []
    y_train = []
    for element in dataset:
        pos_cont = np.concatenate((scaler.fit_transform(element[0].pos), scaler.fit_transform(element[1].pos)), axis = 0).flatten()
        x_train.append(pos_cont)
        y_train.append(element[2].cpu().numpy())
    np.save(f"{name}_x.npy", np.array(x_train)), np.save(f"{name}_y.npy", np.array(y_train))
    return np.array(x_train), np.array(y_train)

if __name__=="__main__":
    train_dataset = ModelNet40_n3(root="data_train")
    test_dataset = ModelNet40_n3(root="data_test")
    scaler = MinMaxScaler()

    x_train, y_train = get_data(train_dataset, "train")
    x_test, y_test = get_data(test_dataset, "test")
    #reg = autosklearn.regression.AutoSklearnRegressor(
        #time_left_for_this_task=120,
        #per_run_time_limit=30,
        #n_jobs=8
    #)
    #reg.fit(x_train, y_train)
    #print(reg.leaderboard())
    #reg.predict(x_test)
    #print(reg.score(x_test, y_test))

    #reg = xgb.XGBRegressor(tree_method="hist", early_stopping_rounds=10)
    reg = MultiOutputRegressor(GradientBoostingRegressor(random_state=0, verbose=True, learning_rate=1, n_iter_no_change=5, max_depth=10, n_estimators=100), n_jobs = 8)
    #reg = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=0, verbose=True), n_jobs = 14)
    #reg = MultiOutputRegressor(AdaBoostRegressor(random_state=0, verbose=True), n_jobs = 14)
    print("Creating regressor")
    #clf = GridSearchCV(
        #reg,
        #{"max_depth": [2, 3, 4, 5, 6, 7, 8], "n_estimators": [100]},
        #verbose=1,
        #n_jobs=4,
    #)
    #clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)
    #print(clf.best_score_)
    #print(clf.best_params_)
    #print(clf.best_estimator_)
    #reg.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)
    reg.fit(x_train, y_train)
    print(np.average(np.abs(reg.predict(x_test) - y_test), axis=0))
    print(reg.score(x_test, y_test))