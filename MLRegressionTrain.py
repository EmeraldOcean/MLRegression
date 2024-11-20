import warnings
warnings.filterwarnings('ignore')

import numpy as np
import datetime
from MLRegressionModel import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold


class PrepareData:
    def __init__(self, dataPath, date=True, scaling=False, dateName=None, startMinute=None, endMinute=None, rate=(8,2,0)):
        self.dataPath = dataPath  # training data path
        self.date = date  # exist of date column
        self.scaling = scaling
        self.dateName = dateName  # name of date column
        self.startMinute = startMinute
        self.endMinute = endMinute
        self.rate = rate

        self.fileList = os.listdir(self.dataPath)
        self.Xcolumn = list(range(4))  # column list for X
        self.ycolumn = [5]  # column list for y

    def loadData(self):
        X = np.empty((0, len(self.Xcolumn)))
        y = np.empty((0, len(self.ycolumn)))
        for i, name in enumerate(self.fileList):
            df = pd.read_csv(f"{self.dataPath}/{name}")

            if "Unnamed: 0" in df.columns:
                df.drop(['Unnamed: 0'], axis=1, inplace=True)  # remove column
            if self.date:
                df[self.dateName] = pd.to_datetime(df[self.dateName])

            start_day = df[self.dateName][0]
            date = datetime.datetime(start_day.year, start_day.month, start_day.day, start_day.hour, start_day.minute, start_day.second)

            start = date + datetime.timedelta(minutes=self.startMinute)
            end = date + datetime.timedelta(minutes=self.endMinute)
            train_df = df[df[self.dateName].between(start, end)]

            # concatenate with X, y data
            Xdata = train_df[self.Xcolumn].to_numpy()
            ydata = train_df[self.ycolumn].to_numpy()

            X = np.append(X, Xdata, axis=0)
            y = np.append(y, ydata, axis=0)
        return X, y

    def splitData(self):
        np.random.seed(42)
        num_data = len(self.Xcolumn)
        train_rate = self.rate[0]
        test_rate = self.rate[1]
        valid_rate = self.rate[2]
        num_train, num_test = int(train_rate * num_data), int(test_rate * num_data)

        random_idx = np.random.permutation(num_data)

        train_idx, test_valid_idx = random_idx[:num_train], random_idx[num_train:]

        X, y = self.loadData()
        if self.scaling:
            X, y = self.scalingData(X, y)

        if valid_rate == 0:
            test_idx = test_valid_idx
            (x_train, x_test, y_train, y_test) = (X[train_idx], X[test_idx], y[train_idx], y[test_idx])
            return (x_train, x_test, y_train, y_test)
        else:
            test_idx, valid_idx = (test_valid_idx[:num_test], test_valid_idx[num_test:])
            (x_train, x_test, x_valid, y_train, y_test, y_valid) = (X[train_idx], X[test_idx], X[valid_idx], y[train_idx], y[test_idx], y[valid_idx])
            return (x_train, x_test, x_valid, y_train, y_test, y_valid)

    def scalingData(self, X, y):
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaled_x = scaler_x.fit_transform(X)
        scaled_y = scaler_y.fit_transform(y)
        return scaled_x, scaled_y


class MLRegressionTrain:
    def __init__(self, X, y, modelName, configPath, savePath, kFold = False):
        self.X = X
        self.y = y
        self.modelName = modelName
        self.configPath = configPath
        self.savePath = savePath
        self.kFold = kFold

        if self.modelName == "LinearRegression":
            self.model = _LinearRegression(self.configPath)

        elif self.modelName == "LassoRegression":
            self.model = _LassoRegression(self.configPath)

        elif self.modelName == "RidgeRegression":
            self.model = _RidgeRegression(self.configPath)

        elif self.modelName == 'ElasticNet':
            self.model = _ElasticNet(self.configPath)

        elif self.modelName == 'MultiTaskElasticNet':
            self.model = _MultiTaskElasticNet(self.configPath)

        elif self.modelName == "PolynomialRegression":
            self.model = _PolynomialRegression(self.configPath)

        elif self.modelName == "RandomForestRegression":
            self.model = _RandomForestRegression(self.configPath)

        elif self.modelName == "KNNRegression":
            self.model = _KNNRegression(self.configPath)

        elif self.modelName == "XGBoost":
            self.model = _XGBoost(self.configPath)

        else:
            raise ValueError("Unknown model")


    # K-Fold Train
    def KFoldTrain(self, **kwargs):
        best_model = None
        bestMSE = 999
        mse_list, mae_list, r2_score_list = [], [], []
        kfold = KFold(**kwargs)
        for train_index, test_index in kfold.split(self.X):
            x_train, x_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            new_model = self.model
            new_model.fit(x_train, y_train)
            y_pred = new_model.predict(x_test)
            mse_list.append(mean_squared_error(y_test, y_pred))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            r2_score_list.append(r2_score(y_test, y_pred))

            tmpMSE = mean_squared_error(y_test, y_pred)
            print(tmpMSE)

            # MSE 기준으로 최적의 모델 선택
            if tmpMSE < bestMSE:
                best_model = new_model
                bestMSE = tmpMSE
        joblib.dump(best_model, "%s/%s.pkl" % (self.savePath, self.modelName))

        return np.mean(mse_list), np.mean(mae_list), np.mean(r2_score_list)

    def train(self, X_train, X_test, y_train, y_test, **kwargs):
        if self.kFold:
            return self.KFoldTrain(**kwargs)
        else:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            joblib.dump(self.model, "%s/%s.pkl" % (self.savePath, self.modelName))

            return mse, mae, r2


if __name__ == "__main__":
    prepareData = PrepareData("(data_path)", True, False, "(Date)", 0, 60)
    X_train, X_test, y_train, y_test = prepareData.splitData()
    model_list = ["LinearRegression", "LassoRegression", "RidgeRegression", "ElasticNet", "MultiTaskElasticNet", "PolynomialRegression", "RandomForestRegression", "KNNRegression", "XGBoost"]

    for model in model_list:
        trainer = MLRegressionTrain(X, y, model, "MLRegressionConfig.json", "(savePath)")
        mse, mae, r2 = trainer.train(X_train, X_test, y_train, y_test)
        print(f"model: {model}, mse: {mse: 0.4f}, mae: {mae: 0.4f}, r2: {r2: 0.4f}")