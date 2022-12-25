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
    """
    This class shows some information about the dataset
    """

    def __init__(self, df):

        self.df = df
        print()
        print('Information object is created')
        print()

    def details(self, df):
        """
        desc: This function finds the missing values in the dataset
        :param df: DataFrame- The data you to wanna see the details about.
        :return: return the missing values in  descending order.
        """
        # get the sum of all missing values in the dataset
        self.missing_values = df.isnull().sum()
        # sorting the missing values in a pandas Series
        self.missing_values = self.missing_values.sort_values(ascending=False)

        # returning the missing values Series
        print("=" * 50)
        print("Missing Values in the dataset. ====>  ", self.missing_values)
        print("=" * 50)
        print()
        # Skewness in the dataset.
        self.skewed_data = self.df.skew()
        self.skewed_data = self.skewed_data.sort_values(ascending=False)
        print("=" * 50)
        print("Skewed Columns in the dataset. ====>  ", self.skewed_data)
        print("=" * 50)
        print()
        feature_dtypes = self.df.dtypes
        print("=" * 50)
        print("Data types of the dataset. ====>  ", feature_dtypes)
        print("=" * 50)
        print()

        rows, columns = self.df.shape
        print("=" * 50)
        print('====> This data contains {} rows and {} columns'.format(rows, columns))
        print("=" * 50)
        print()

        correlation = self.df.corr()
        print("=" * 50)
        print('====> This is the correlation of dataset  ', correlation)
        print("=" * 50)
        print()



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
        """
        desc: This tells us the best predictor between cat vs con
        :param df: Dataset
        :param cat: Catgorical data
        :param con: Continous Data
        :return: Return the value of f1 which shows the best predictor on the basis of anova model
        """
        rel = con + " ~ " + cat
        model = ols(rel, df).fit()

        anova_results = anova_lm(model)
        Q = DataFrame(anova_results)
        a = Q['PR(>F)'][cat]
        return round(a, 3)

    def preprocessing(self, df):
        """
        desc: This method is for  preprocessing the data means it will convert the categorical data
            into continous. After doing cat to con we will convert the data into standardized format.
        :param df: Passing the data set
        :return: It returns standardized data.
        """
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
        """
        :desc: This is the method which shows model builder which can be easily.
        :param df: DataFrame contains whole data from your datasets.
        :param Ycol:Main Predicted columns Eg: Price, Profit, Sales, Eligible etc.
        :param cols_to_drop: Those columns which are not capable to give best model.
        :param model_obj: Model object contains like decision Tree, Random Forest etc.
        :return: Return the model is overfitting or not.
        """

        import pandas as pd
        df = df.drop(labels=cols_to_drop, axis=1)
        re = preprocessing()
        self.replacer(df)
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
        """
        Desc: Model Tuning using the Params like max depth, kneighbours, alpha etc. To improve the model accuracy.
        :param df: DataFrame contains whole data from your datasets.
        :param Ycol: Main predicted column.
        :param cols_to_drop: THose columns which doesnt give value or improvement to your model
        :param model_obj: Model Object Like Decision Tree, Random Forest, Adaboost, SVM, etc.
        :param tp: Tuning Parameter Like n_estimators, max_depth, min_sample_Split or leaf, alpha etc.
        :return: It returns the best parameter which you can pass with your model and make the better one.
        """
        df = df.drop(labels=cols_to_drop, axis=1)
        re = preprocessing()
        self.replacer(df)
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

    def Univariate(self, df, figsize, rows, columns):
        """
        desc: Univariate method it means only analysis of individual columns.
        :param df: df is the dataset contains all columns.
        :param figsize: You have to pass the figure size of a chart eg. (20, 25)
        :param rows: You can specify rows. To show your charts in the good form.
            This depends on your data eg. data contains 15 columns so you can pass 5
        :param columns: You can specify Clumns to show the charts
            This depends on your data eg. data contains 15 columns so you can pass 3
        :return: It returns the countplot and distplot.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        x = 1
        plt.figure(figsize=figsize)
        for i in df.columns:
            if df[i].dtypes == 'object':
                plt.subplot(rows, columns, x)
                sns.countplot(df[i])
                x = x + 1
            else:
                plt.subplot(rows, columns, x)
                sns.distplot(df[i])
                x = x + 1

    def Bivariate(self, df, Y, figsize, rows, columns):
        """
        desc: Bivariate method it shows  analysis between whole data vs main predicted columns.
        :param df: df is the dataset contains all columns.
        :param Y:
        :param figsize: You have to pass the figure size of a chart eg. (25, 35)
        :param rows: You can specify rows. To show your charts in the good form.
            This depends on your data eg. data contains 12 columns so you can pass 4
        :param columns: You can specify Clumns to show the charts
            This depends on your data eg. data contains 12 columns so you can pass 3
        :return: Return two kind of charts 1. Scatter, and 2. Boxplot
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        x = 1
        plt.figure(figsize=figsize)
        for i in df.columns:
            if df[i].dtypes == 'object':
                plt.subplot(rows, columns, x)
                sns.boxplot(df[i], df[Y])
                x = x + 1
            else:
                plt.subplot(rows, columns, x)
                sns.scatterplot(df[i], df[Y])
                x = x + 1


class MLModels:
    def linearModel(self, x, y):
        """
        desc: This is Linear Model Method Directly You can create it without showing the error.
        :param x: This is an training data, contains all collumns to predict main columns eg. RND, MKT ~ PROFIT
        :param y: In this we will pass training data, of main predicted columns eg. PROFIT
        :return: IT return the model that you can use for further prediction
        """
        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()  # lm --> model object
        model = lm.fit(x, y)
        return model

    def lassoModel(self, x, y):
        """
        desc: This is Lasso Model Method Directly You can create it without showing the error.
        :param x: This is an training data, contains all collumns to predict main columns eg. RND, MKT ~ PROFIT
        :param y: In this we will pass training data, of main predicted columns eg. PROFIT
        :return: IT return the model that you can use for further prediction
        """
        from sklearn.linear_model import Lasso, LassoCV
        lscv = LassoCV(alphas=None, max_iter=100000, normalize=True)  # lm --> model object
        modelcv = lscv.fit(x, y)
        ls = Lasso(alpha=modelcv.alpha_)
        model = ls.fit(x, y)
        return model

    def RidgeModel(self, x, y):
        """
        desc: This is Ridge Model Method Directly You can create it without showing the error.
        :param x: This is an training data, contains all collumns to predict main columns eg. RND, MKT ~ PROFIT
        :param y: In this we will pass training data, of main predicted columns eg. PROFIT
        :return: IT return the model that you can use for further prediction
        """
        from sklearn.linear_model import Ridge, RidgeCV
        rcv = RidgeCV(alphas=None)  # lm --> model object
        modelcv = rcv.fit(x, y)
        ls = Ridge(alpha=modelcv.alpha_)
        model = ls.fit(x, y)
        return model

    def sweetVizEda(self, df, Ycol):
        """
        desc: This is an sweet viz library to show the charts and visuals for your data.
        :param df: Data set
        :param Ycol: Main Predicted column
        :return: Return the html to see the charts and visuals in one go.
        """
        import sweetviz
        my_report = sweetviz.analyze([df, 'Train'], target_feat=Ycol)
        return my_report.show_html('EDAReport.html')

    def sweetVizEdaTrTs(self, train, test, Ycol):
        """
        desc: Sweet VIZ EDA Training Testing Report . That also we can make for comparision  of data BW training and testing
        :param train: Pass the whole testing data for you training visuals, charts and eda.
        :param test: Pass the whole testing data for you testing visuals, charts and eda.
        :param Ycol: Pass the main predicted columns eg. "SalePrice"
        :return: Return the Comparision Report.
        """
        import sweetviz
        my_report1 = sweetviz.compare([train, 'Train'], [test, 'Test'], Ycol)
        return my_report1.show_html('Comparision_Report.html')

    def RndmFrstClsModel(self, X, Y):
        """
        desc: This method is Random Forest Classsfication model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        desc: This method is Random Forest Regression model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        DESC: This is the Random Forest model which automatically detect your data and apply the apppropriate model.
            For eg. your y is continous so this will apply the regression model without specifying it.
        :param X: Only Preprocessed data you have to pass for X data.
        :param Y: Here you have to pass the Y column.
        :return: Automatically detects the model is regression or classification.
            On the basis of the Column it will return the classification or regression.
        """
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.RndmFrstClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.RndmFrstRgrModel(X, Y)

    def AdaBostClsModel(self, X, Y):
        """
        desc: This method is AdaBoost Classsfication model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        desc: This method is AdaBoost Regression model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        DESC: This is the AdaBOOST model which automatically detect your data and apply the apppropriate model.
            For eg. your y is continous so this will apply the regression model without specifying it.
        :param X: Only Preprocessed data you have to pass for X data.
        :param Y: Here you have to pass the Y column.
        :return: Automatically detects the model is regression or classification.
            On the basis of the Column it will return the classification or regression.
        """
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.AdaBostClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.AdaBostRgrModel(X, Y)

    def DcsnTreeClsModel(self, X, Y):
        """
        desc: This method is Decision Tree Classsfication model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        desc: This method is Decision Tree Regression model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        DESC: This is the Decision Tree model which automatically detect your data and apply the apppropriate model.
            For eg. your y is continous so this will apply the regression model without specifying it.
        :param X: Only Preprocessed data you have to pass for X data.
        :param Y: Here you have to pass the Y column.
        :return: Automatically detects the model is regression or classification.
            On the basis of the Column it will return the classification or regression.
        """
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.DcsnTreeClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.DcsnTreeRgrModel(X, Y)

    def KNNClsModel(self, X, Y):
        """
        desc: This method is KNearest Neighbor Classsfication model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        desc: This method is KNearest Neighbor Regression model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        DESC: This is the KNN model which automatically detect your data and apply the apppropriate model.
            For eg. your y is continous so this will apply the regression model without specifying it.
        :param X: Only Preprocessed data you have to pass for X data.
        :param Y: Here you have to pass the Y column.
        :return: Automatically detects the model is regression or classification.
            On the basis of the Column it will return the classification or regression.
        """
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.KNNClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.KNNRegModel(X, Y)

    def SVMClsModel(self, X, Y):
        """
        desc: This method is Support Vector Classsfication model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_score etc.
        """
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
        svc = SVC(C=bst['C'], gamma=bst['gamma'], kernel=bst['kernel'])
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
            print(f1_score(Y, pred, average='macro'), f1_score(Y, pred, average='micro'),
                  f1_score(Y, pred, average='weighted'), f1_score(Y, pred, average=None))
        return model, accuracy_score(Y, pred)

    def SVMRegModel(self, X, Y):
        """
        desc: This method is Support Vector Regression model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_score etc.
        """
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
        svr = SVR(C=bst['C'], gamma=bst['gamma'], kernel=bst['kernel'])
        model = svr.fit(X, Y)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        pred = model.predict(X)
        print("MeanAbsError", mean_absolute_error(Y, pred), end="\n")
        return model

    def SVMModel(self, X, Y):
        """
        DESC: This is the SVM model which automatically detect your data and apply the apppropriate model.
            For eg. your y is continous so this will apply the regression model without specifying it.
        :param X: Only Preprocessed data you have to pass for X data.
        :param Y: Here you have to pass the Y column.
        :return: Automatically detects the model is regression or classification.
            On the basis of the Column it will return the classification or regression.
        """
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.SVMClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.SVMRegModel(X, Y)

    def XGBoostClsModel(self, X, Y):
        """
        desc: This method is XGBoost Classsfication model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        desc: This method is XGBoost Regression model. Using this model you can do multiple things.
             Like Tuning, Get best Param, accuracy, f1 score, Label Encoding etc.
        :param X: Pass the Preprocessed data contains all columns except Y columns.
        :param Y: Pass the Y column only
        :return: Return the F1 Score, Error, model, accuracy_scoreetc.
        """
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
        """
        DESC: This is the XGBOOST model which automatically detect your data and apply the apppropriate model.
            For eg. your y is continous so this will apply the regression model without specifying it.
        :param X: Only Preprocessed data you have to pass for X data.
        :param Y: Here you have to pass the Y column.
        :return: Automatically detects the model is regression or classification.
            On the basis of the Column it will return the classification or regression.
        """
        if Y.dtypes[0] == "object":
            mlmdl = MLModels()
            return mlmdl.XGBoostClsModel(X, Y)
        else:
            mlmdl = MLModels()
            return mlmdl.XGBoostRgrModel(X, Y)

    def overAllModelsClsAccuracy(self, xtrain, ytrain, xtest, ytest):
        """
        desc: This method is Over all models of classification accuracy.
            This just compare your models accuracy and show the best model.
        :param xtrain: Pass the Preprocessed training data except ytrain columns.
        :param ytrain: Pass the ytrain column only
        :param xtest: Pass the Preprocessed testing data except ytest columns.
        :param ytest: Pass the ytest column only
        :return: Return the Comparision bw all the model and return the best model.
        """
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
        """
        desc: Test MOdel this will make your model best on the bassis of test model you can see the model is overfitting or not.
        :param X: Just have to pass the test data.
        :param model: Pass the model like Decision tree, Random Forest etc.
        :return: THe accuracy or error of the test model.
        """
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
        """
        desc: Test model Comparision Here We have to pass the X and Y along with model.
        :param X: Whole Preprocessed test data.
        :param Y: Pass the Y columns
        :param model: Eg. Like Decision tree, random Forest, Linear regreassion , Logistic regression etc.
        :return: REturn the comparision bw the actual and predicted data.
        """
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
