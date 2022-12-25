import seaborn as sns
import matplotlib.pyplot as plt


class Visualized:

    def Univariate(self, A, figsize, rows, columns):
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