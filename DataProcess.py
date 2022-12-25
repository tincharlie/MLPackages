import pandas  as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


class PreProcessing:
    def replacer(self, df):
        """
        desc: This is the replacer function for our data. To replace the missing values in dataset.
        :param df: Pass the data to fill all the missing values.
        :return: It will replace all the missing values on the basis of mode and mean.
        """
        import pandas as pd
        Q = pd.DataFrame(df.isna().sum(), columns=["ct"])
        for i in Q[Q.ct > 0].index:
            if df[i].dtypes == "object":
                x = df[i].mode()[0]
                df[i] = df[i].fillna(x)
            else:
                x = df[i].mean()
                df[i] = df[i].fillna(x)

    def ANOVA(self, df, cat, con):

        rel = con + " ~ " + cat
        model = ols(rel, df).fit()

        anova_results = anova_lm(model)
        Q = pd.DataFrame(anova_results)
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
        re = PreProcessing()
        self.replacer(df)
        Y = df[Ycol]
        X = df.drop(labels=Ycol, axis=1)
        X_new = PreProcessing(X)
        from sklearn.model_selection import train_test_split
        xtrain, xtest, ytrain, ytest = train_test_split(X_new, Y, test_size=0.2, random_state=31)
        if (ytrain[Ycol[0]].dtypes == "object"):
            re.find_overfit_cat(model_obj, xtrain, xtest, ytrain, ytest)
        else:
            re.find_overfit_con(model_obj, xtrain, xtest, ytrain, ytest)

    def CV_tune(self, df, Ycol, cols_to_drop, model_obj, tp):
        df = df.drop(labels=cols_to_drop, axis=1)
        re = PreProcessing()
        self.replacer(df)
        Y = df[Ycol]
        X = df.drop(labels=Ycol, axis=1)
        X_new = PreProcessing(X)
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