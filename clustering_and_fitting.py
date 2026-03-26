"""
Clustering and Fitting Assignment
Student Name: Sukesh Kumar Eddagiri
Student ID: 25036788
Dataset: data.csv(Mall_Customers)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import warnings


def plot_relational_plot(df):
    fig, ax = plt.subplots()
    ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Annual Income vs Spending Score")
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    fig, ax = plt.subplots()
    ax.hist(df['Age'], bins=15)
    ax.set_xlabel("Age")
    ax.set_title("Age Distribution")
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    fig, ax = plt.subplots()
    sns.heatmap(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr(),
                annot=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    print("Head:\n", df.head())
    print("\nDescription:\n", df.describe())
    print("\nCorrelation:\n", df.corr(numeric_only=True))
    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    if moments[2] > 0:
        skew_desc = "Right_Skewed"
    elif moments[2] < 0:
        skew_desc = "Left_Skewed"
    else:
        skew_desc = "Not_Skewed"

    if moments[3] > 0:
        kurt_desc = "Leptokurtic"
    elif moments[3] < 0:
        kurt_desc = "Platykurtic"
    else:
        kurt_desc = "Mesokurtic"

    print(f'The data was {skew_desc} and {kurt_desc}.')
    return


def perform_clustering(df, col1, col2):

    def plot_elbow_method():
        inertias = []
        k_range = range(1, 10)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(k_range, inertias, marker='o')
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        plt.savefig('elbow_plot.png')
        plt.close()
        return

    def one_silhouette_inertia():
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(data)
        _score = None  # not calculated in notebook
        _inertia = kmeans.inertia_
        return _score, _inertia

    # Ignore Warnings
    warnings.filterwarnings("ignore")
   

    # Gather data
    data = df[[col1, col2]]

    # Fit KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(data)

    # Elbow + inertia
    one_silhouette_inertia()
    plot_elbow_method()

    # Cluster centers
    centres = kmeans.cluster_centers_
    xkmeans = centres[:, 0]
    ykmeans = centres[:, 1]
    cenlabels = range(len(centres))

    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    fig, ax = plt.subplots()
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels)
    ax.scatter(xkmeans, ykmeans, c='red', marker='x', s=100)
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.set_title("K-Means Clustering")
    plt.savefig('clustering.png')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    X = df[[col1]]
    y = df[col2]

    model = LinearRegression()
    model.fit(X, y)

    x = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x)

    return df, x, y_pred


def plot_fitted_data(data, x, y):
    fig, ax = plt.subplots()
    ax.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
    ax.plot(x, y)
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Linear Regression Fit")
    plt.savefig('fitting.png')
    plt.close()
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    col = 'Age'

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)

    clustering_results = perform_clustering(
        df, 'Annual Income (k$)', 'Spending Score (1-100)'
    )
    plot_clustered_data(*clustering_results)

    fitting_results = perform_fitting(
        df, 'Annual Income (k$)', 'Spending Score (1-100)'
    )
    plot_fitted_data(*fitting_results)

    return


if __name__ == '__main__':
    main()
