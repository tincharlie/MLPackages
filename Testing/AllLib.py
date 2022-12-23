from pandas import DataFrame
from statsmodels.api import OLS
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score


class preprocessing:
    def replacer(self, df):
        import pandas as pd
        Q = pd.DataFrame(df.isna().sum(), columns=["ct"])
        for i in Q[Q.ct > 0].index:
            if (df[i].dtypes == "object"):
                x = df[i].mode()[0]
                df[i] = df[i].fillna(x)
            else:
                x = df[i].mean()
                df[i] = df[i].fillna(x)

    def ANOVA(self, df, cat, con):

        rel = con + " ~ " + cat
        model = ols(rel, df).fit()

        anova_results = anova_lm(model)
        Q = DataFrame(anova_results)
        a = Q['PR(>F)'][cat]
        return round(a, 3)

    def preprocessing(self, df):

        cat = []
        con = []
        for i in df.columns:
            if (df[i].dtypes == "object"):
                cat.append(i)
            else:
                con.append(i)
        X1 = pd.get_dummies(df[cat])

        ss = StandardScaler()
        X2 = pd.DataFrame(ss.fit_transform(df[con]), columns=con)
        X3 = X2.join(X1)
        return X3

    def find_overfit_cat(self, model_obj, xtrain, xtest, ytrain, ytest):
        model = model_obj.fit(xtrain, ytrain)
        pred_ts = model.predict(xtest)
        pred_tr = model.predict(xtrain)
        from sklearn.metrics import accuracy_score
        print("training Accuracy: ", accuracy_score(ytrain, pred_tr))
        print("testing Accuracy: ", accuracy_score(ytest, pred_ts))

    def find_overfit_con(self, model_obj, xtrain, xtest, ytrain, ytest):
        model = model_obj.fit(xtrain, ytrain)
        pred_ts = model.predict(xtest)
        pred_tr = model.predict(xtrain)
        from sklearn.metrics import mean_absolute_error
        print("training error: ", mean_absolute_error(ytrain, pred_tr))
        print("testing error: ", mean_absolute_error(ytest, pred_ts))

    def model_builder(self, df, Ycol, cols_to_drop, model_obj):
        import pandas as pd
        df = df.drop(labels=cols_to_drop, axis=1)
        re = preprocessing()
        re.replacer(df)
        Y = df[Ycol]
        X = df.drop(labels=Ycol, axis=1)
        X_new = preprocessing(X)
        from sklearn.model_selection import train_test_split
        xtrain, xtest, ytrain, ytest = train_test_split(X_new, Y, test_size=0.2, random_state=31)
        if (ytrain[Ycol[0]].dtypes == "object"):
            re.find_overfit_cat(model_obj, xtrain, xtest, ytrain, ytest)
        else:
            re.find_overfit_con(model_obj, xtrain, xtest, ytrain, ytest)

    def CV_tune(self, df, Ycol, cols_to_drop, model_obj, tp):
        import pandas as pd
        df = df.drop(labels=cols_to_drop, axis=1)
        re = preprocessing()
        re.replacer(df)
        Y = df[Ycol]
        X = df.drop(labels=Ycol, axis=1)
        X_new = preprocessing(X)
        from sklearn.model_selection import train_test_split
        xtrain, xtest, ytrain, ytest = train_test_split(X_new, Y, test_size=0.2, random_state=31)
        if (ytrain[Ycol[0]].dtypes == "object"):
            from sklearn.model_selection import GridSearchCV
            cv = GridSearchCV(model_obj, tp, scoring="accuracy", cv=4)
            cvmodel = cv.fit(xtrain, ytrain)
            print(cvmodel.best_params_)
        else:
            from sklearn.model_selection import GridSearchCV
            cv = GridSearchCV(model_obj, tp, scoring="neg_mean_absolute_error", cv=4)
            cvmodel = cv.fit(xtrain, ytrain)
            print(cvmodel.best_params_)

    def Univariate(self, A, figsize, rows, columns):
        import seaborn as sns
        import matplotlib.pyplot as plt
        x = 1
        plt.figure(figsize=figsize)
        for i in A.columns:
            if A[i].dtypes == 'object':
                plt.subplot(rows, columns, x)
                sns.countplot(A[i])
                x = x + 1
            else:
                plt.subplot(rows, columns, x)
                sns.distplot(A[i])
                x = x + 1

    def Bivariate(self, A, Y, figsize, rows, columns):
        import seaborn as sns
        import matplotlib.pyplot as plt
        x = 1
        plt.figure(figsize=figsize)
        for i in A.columns:
            if A[i].dtypes == 'object':
                plt.subplot(rows, columns, x)
                sns.boxplot(A[i], A[Y])
                x = x + 1
            else:
                plt.subplot(rows, columns, x)
                sns.scatterplot(A[i], A[Y])
                x = x + 1


class MLModels:
    def linearModel(self, x, y):
        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()  # lm --> model object
        model = lm.fit(x, y)
        return model

    def lassoModel(self, x, y):
        from sklearn.linear_model import Lasso, LassoCV
        lscv = LassoCV(alphas=None, max_iter=100000, normalize=True)  # lm --> model object
        modelcv = lscv.fit(x, y)
        ls = Lasso(alpha=modelcv.alpha_)
        model = ls.fit(x, y)
        return model

    def RidgeModel(self, x, y):
        from sklearn.linear_model import Ridge, RidgeCV
        rcv = RidgeCV(alphas=None)  # lm --> model object
        modelcv = rcv.fit(x, y)
        ls = Ridge(alpha=modelcv.alpha_)
        model = ls.fit(x, y)
        return model

    def sweetVizEda(self, df, Ycol):
        import sweetviz
        my_report = sweetviz.analyze([df, 'Train'], target_feat=Ycol)
        return my_report.show_html('EDAReport.html')

    def sweetVizEdaTrTs(self, train, test, Ycol):
        import sweetviz
        my_report1 = sweetviz.compare([train, 'Train'], [test, 'Test'], 'SalePrice')
        return my_report1.show_html('Comparision_Report.html')

    def RndmFrstClsModel(self, X, Y):
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(random_state=40)
        tp = {'n_estimators': range(1, 40, 1)}
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(rfc, tp, scoring='accuracy', cv=5)
        gcv = cv.fit(X, Y)
        bst = gcv.best_params_
        print(bst)
        rfc = RandomForestClassifier(random_state=40, n_estimators=bst['n_estimators'])
        model = rfc.fit(X, Y)
        from sklearn.metrics import accuracy_score, f1_score
        pred = model.predict(X)
        print("accuracy_score", accuracy_score(Y, pred), end="\n")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        pred = le.fit_transform(pred)
        print("f1 score", f1_score(Y, pred))
        return model, accuracy_score(Y, pred)

    def RndmFrstRgrModel(self, X, Y):
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(random_state=40)
        dtp = {'n_estimators': range(1, 40, 1)}
        from sklearn.model_selection import GridSearchCV
        dcv = GridSearchCV(rfr, dtp, scoring='neg_mean_absolute_error', cv=4)
        dgcv = dcv.fit(X, Y)
        dbst = dgcv.best_params_
        print(dbst)
        rfr = RandomForestRegressor(random_state=42, n_estimators=dbst['n_estimators'])
        model = rfr.fit(X, Y)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        pred = model.predict(X)
        print("MeanAbsError", mean_absolute_error(Y, pred), end="\n")
        print("MeanSqrError", mean_squared_error(Y, pred), end="\n")
        return model

    def RndmFrstModel(self, X, Y):
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.RndmFrstClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.RndmFrstRgrModel(X, Y)

    def AdaBostClsModel(self, X, Y):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier(random_state=40)
        abc = AdaBoostClassifier(dtc, random_state=42)
        tp = {'n_estimators': range(1, 40, 1)}
        dtp = {'max_depth': range(1, 40, 1)}
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(abc, tp, scoring='accuracy', cv=4)
        dcv = GridSearchCV(dtc, dtp, scoring='accuracy', cv=4)
        gcv = cv.fit(X, Y)
        dgcv = dcv.fit(X, Y)
        bst = gcv.best_params_
        dbst = dgcv.best_params_
        print(bst)
        print(dbst)
        dtc = DecisionTreeClassifier(random_state=40, max_depth=dbst['max_depth'])
        abr = AdaBoostClassifier(dtc, random_state=42, n_estimators=bst['n_estimators'])
        model = abr.fit(X, Y)
        from sklearn.metrics import accuracy_score, f1_score
        pred = model.predict(X)
        print("accuracy_score", accuracy_score(Y, pred), end="\n")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        pred = le.fit_transform(pred)
        print("f1 score", f1_score(Y, pred))
        return model, accuracy_score(Y, pred)

    def AdaBostRgrModel(self, X, Y):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        dtr = DecisionTreeRegressor(random_state=40)
        abr = AdaBoostRegressor(dtr, random_state=42)
        tp = {'n_estimators': range(1, 40, 1)}
        dtp = {'max_depth': range(1, 40, 1)}
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(abr, tp, scoring='neg_mean_absolute_error', cv=4)
        dcv = GridSearchCV(dtr, dtp, scoring='neg_mean_absolute_error', cv=4)
        gcv = cv.fit(X, Y)
        dgcv = dcv.fit(X, Y)
        bst = gcv.best_params_
        dbst = dgcv.best_params_
        print(bst)
        print(dbst)
        dtr = DecisionTreeRegressor(random_state=42, max_depth=dbst['max_depth'])
        # abc = AdaBoostClassifier(dtc, random_state=21,n_estimators=20)
        abr = AdaBoostRegressor(dtr, random_state=42, n_estimators=bst['n_estimators'])
        model = abr.fit(X, Y)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        pred = model.predict(X)
        print("MeanAbsError", mean_absolute_error(Y, pred), end="\n")
        print("MeanSqrError", mean_squared_error(Y, pred), end="\n")
        return model

    def AdaBoostModel(self, X, Y):
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.AdaBostClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.AdaBostRgrModel(X, Y)

    def DcsnTreeClsModel(self, X, Y):
        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier(random_state=40)
        dtp = {'max_depth': range(1, 40, 1)}
        from sklearn.model_selection import GridSearchCV
        dcv = GridSearchCV(dtc, dtp, scoring='accuracy', cv=4)
        dgcv = dcv.fit(X, Y)
        dbst = dgcv.best_params_
        print(dbst)
        dtc = DecisionTreeClassifier(random_state=40, max_depth=dbst['max_depth'])
        model = dtc.fit(X, Y)
        from sklearn.metrics import accuracy_score, f1_score
        pred = model.predict(X)
        print("accuracy_score", accuracy_score(Y, pred), end="\n")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        pred = le.fit_transform(pred)
        print("f1 score", f1_score(Y, pred))
        return model, accuracy_score(Y, pred)

    def DcsnTreeRgrModel(self, X, Y):
        from sklearn.tree import DecisionTreeRegressor
        dtr = DecisionTreeRegressor(random_state=40)
        dtp = {'max_depth': range(1, 40, 1)}
        from sklearn.model_selection import GridSearchCV
        dcv = GridSearchCV(dtr, dtp, scoring='neg_mean_absolute_error', cv=4)
        dgcv = dcv.fit(X, Y)
        dbst = dgcv.best_params_
        print(dbst)
        dtr = DecisionTreeRegressor(random_state=42, max_depth=dbst['max_depth'])
        model = dtr.fit(X, Y)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        pred = model.predict(X)
        print("MeanAbsError", mean_absolute_error(Y, pred), end="\n")
        print("MeanSqrError", mean_squared_error(Y, pred), end="\n")
        return model

    def DcsnTreeModel(self, X, Y):
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.DcsnTreeClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.DcsnTreeRgrModel(X, Y)

    def KNNClsModel(self, X, Y):
        from sklearn.neighbors import KNeighborsClassifier
        knc = KNeighborsClassifier()
        tp = {'n_neighbors': range(1, 40), "weights": ['distance', 'uniform']}
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(knc, tp, scoring='accuracy', cv=4)
        gcv = cv.fit(X, Y)
        bst = gcv.best_params_
        print(gcv.best_params_, end="\n")
        knc = KNeighborsClassifier(n_neighbors=bst['n_neighbors'], weights=bst['weights'])
        model = knc.fit(X, Y)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        pred = model.predict(X)
        print("accuracy_score", accuracy_score(Y, pred), end="\n")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        pred = le.fit_transform(pred)
        print("f1 score", f1_score(Y, pred))
        return model, accuracy_score(Y, pred)

    def KNNRegModel(self, X, Y):
        from sklearn.neighbors import KNeighborsClassifier
        knc = KNeighborsClassifier()
        tp = {'n_neighbors': range(1, 40), "weights": ['distance', 'uniform']}
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(knc, tp, scoring="neg_mean_absolute_error", cv=4)
        gcv = cv.fit(X, Y)
        bst = gcv.best_params_
        print(bst, end="\n")
        knc = KNeighborsClassifier(n_neighbors=bst['n_neighbors'], weights=bst["weights"])
        model = knc.fit(X, Y)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        pred = model.predict(X)
        print("MeanAbsError", mean_absolute_error(Y, pred), end="\n")
        return model

    def KNNModel(self, X, Y):
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.KNNClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.KNNRegModel(X, Y)

    def SVMClsModel(self, X, Y):
        from sklearn.svm import SVC
        svc = SVC()
        tp = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(svc, tp, scoring='accuracy', cv=4)
        gcv = cv.fit(X, Y)
        bst = gcv.best_params_
        print(gcv.best_params_, end="\n")
        svc = SVC(C=bst['C'], gamma = bst['gamma'], kernel=bst['kernel'])
        model = svc.fit(X, Y)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        pred = model.predict(X)
        print("accuracy_score", accuracy_score(Y, pred), end="\n")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        pred = le.fit_transform(pred)
        Y1 = pd.DataFrame(Y, columns=['Y1'])
        Ycount = len(Y1.value_counts())
        if Ycount == 2:
            print("f1 score", f1_score(Y, pred), "\n Precision Score", precision_score(Y, pred))
        else:
            print(f1_score(Y, pred, average='macro'),f1_score(Y, pred, average='micro'), f1_score(Y, pred, average='weighted'),f1_score(Y, pred, average=None))
        return model, accuracy_score(Y, pred)

    def SVMRegModel(self, X, Y):
        from sklearn.svm import SVR
        svr = SVR()
        tp = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(svr, tp, scoring="neg_mean_absolute_error", cv=4)
        gcv = cv.fit(X, Y)
        bst = gcv.best_params_
        print(bst, end="\n")
        svr = SVR(C=bst['C'], gamma = bst['gamma'], kernel=bst['kernel'])
        model = svr.fit(X, Y)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        pred = model.predict(X)
        print("MeanAbsError", mean_absolute_error(Y, pred), end="\n")
        return model

    def SVMModel(self, X, Y):
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.SVMClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.SVMRegModel(X, Y)


    def XGBoostClsModel(self, X, Y):
        from xgboost import XGBClassifier
        xgc = XGBClassifier(random_state=40)
        tp = {'max_depth': range(1, 15, 1), "min_child_weight": [1, 2, 3, 4, 5]}
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(xgc, tp, scoring='accuracy', cv=4)
        gcv = cv.fit(X, Y)
        bst = gcv.best_params_
        print(bst)
        from xgboost import XGBClassifier
        xgc = XGBClassifier(random_state=40, max_depth=bst['max_depth'], min_child_weight=bst['min_child_weight'])
        model = xgc.fit(X, Y)
        from sklearn.metrics import accuracy_score, f1_score
        pred = model.predict(X)
        print("accuracy_score", accuracy_score(Y, pred), end="\n")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        pred = le.fit_transform(pred)
        print("f1 score", f1_score(Y, pred))
        return model, accuracy_score(Y, pred)

    def XGBoostRgrModel(self, X, Y):
        from xgboost import XGBRegressor
        xgr = XGBRegressor(random_state=40)
        tp = {'max_depth': range(1, 15, 1), "min_child_weight": [1, 2, 3, 4, 5]}
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(xgr, tp, scoring='neg_mean_absolute_error', cv=4)
        gcv = cv.fit(X, Y)
        bst = gcv.best_params_
        print(bst)
        xgr = XGBRegressor(random_state=40, max_depth=bst['max_depth'], min_child_weight=bst['min_child_weight'])
        model = xgr.fit(X, Y)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        pred = model.predict(X)
        print("MeanAbsError", mean_absolute_error(Y, pred), end="\n")
        print("MeanSqrError", mean_squared_error(Y, pred), end="\n")
        return model

    def XGBoostModel(self, X, Y):
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.XGBoostClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.XGBoostRgrModel(X, Y)

    def overAllModelsClsAccuracy(self, xtrain, ytrain, xtest, ytest):
        DTC, DTCTrainAcc = self.DcsnTreeClsModel(xtrain, ytrain)
        RFC, RFCTrainAcc = self.RndmFrstClsModel(xtrain, ytrain)
        XGC, XGCTrainAcc = self.XGBoostClsModel(xtrain, ytrain)
        KNC, KNCTrainAcc = self.KNNClsModel(xtrain, ytrain)
        ABC, ABCTrainAcc = self.AdaBostClsModel(xtrain, ytrain)

        pred_dtc_test = DTC.predict(xtest)
        dtc_test_acc = accuracy_score(ytest, pred_dtc_test)
        pred_rfc_test = RFC.predict(xtest)
        rfc_test_acc = accuracy_score(ytest, pred_rfc_test)
        pred_xgc_test = XGC.predict(xtest)
        xgc_test_acc = accuracy_score(ytest, pred_xgc_test)
        pred_knc_test = KNC.predict(xtest)
        knc_test_acc = accuracy_score(ytest, pred_knc_test)
        pred_abc_test = ABC.predict(xtest)
        abc_test_acc = accuracy_score(ytest, pred_abc_test)

        dtc_test_f1score = f1_score(ytest, pred_dtc_test)
        rfc_test_f1score = f1_score(ytest, pred_rfc_test)
        xgc_test_f1score = f1_score(ytest, pred_xgc_test)
        knc_test_f1score = f1_score(ytest, pred_knc_test)
        abc_test_f1score = f1_score(ytest, pred_abc_test)

        if DTCTrainAcc >= RFCTrainAcc and dtc_test_acc > rfc_test_acc and dtc_test_f1score > rfc_test_f1score:
            print("DTC Training Accuracy", DTCTrainAcc, "\n", "DTC Testing Accuracy", dtc_test_acc, )
            print("dtc_test_f1score", dtc_test_f1score)
            return DTC
        elif RFCTrainAcc >= XGCTrainAcc and rfc_test_acc > xgc_test_acc and rfc_test_f1score > xgc_test_f1score:
            print("RFC Training Accuracy", RFCTrainAcc, "\n", "RFC Testing Accuracy", rfc_test_acc)
            print("rfc_test_f1score", rfc_test_f1score)
            return RFC
        elif XGCTrainAcc >= KNCTrainAcc and xgc_test_acc > knc_test_acc and xgc_test_f1score > knc_test_f1score:
            print("XGC Training Accuracy", XGCTrainAcc, "\n", "XGC Testing Accuracy", xgc_test_acc)
            print("xgc_test_f1score", xgc_test_f1score)
            return XGC
        elif KNCTrainAcc >= ABCTrainAcc and knc_test_acc > abc_test_acc and knc_test_f1score > abc_test_f1score:
            print("KNC Training Accuracy", KNCTrainAcc, "\n", "KNC Testing Accuracy", knc_test_acc)
            print("knc_test_f1score", knc_test_f1score)
            return KNC
        elif ABCTrainAcc >= DTCTrainAcc and abc_test_acc > dtc_test_acc and abc_test_f1score > dtc_test_f1score:
            print("ABC Training Accuracy", ABCTrainAcc, "\n", "ABC Testing Accuracy", abc_test_acc)
            print("abc test f1score", abc_test_f1score)
            return ABC


class TestModel:
    def test_model(self, X, model):
        cat = []
        con = []
        for i in X.columns:
            if X[i].dtypes == 'object':
                cat.append(i)
            else:
                con.append(i)
        mm = MinMaxScaler()
        Xcon1 = pd.DataFrame(mm.fit_transform(X[con]), columns=X[con].columns)
        Xcat1 = pd.get_dummies(X[cat])
        X = Xcon1.join(Xcat1)
        pred = model.predict(X)
        tspred = pd.DataFrame()
        tspred['Prediction'] = pred
        return tspred

    def test_model_comparison(self, X, Y, model):
        cat = []
        con = []
        for i in X.columns:
            if X[i].dtypes == 'object':
                cat.append(i)
            else:
                con.append(i)
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        Xcon1 = pd.DataFrame(mm.fit_transform(X[con]), columns=X[con].columns)
        Xcat1 = pd.get_dummies(X[cat])
        X = Xcon1.join(Xcat1)
        pred = model.predict(X)
        tspred = pd.DataFrame()
        tspred['Prediction'] = pred
        tspred["Actual"] = Y
        return tspred
