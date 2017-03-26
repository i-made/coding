'''

Author: Nikhil Kulkarni
Desc: Capital One coding challenge
Note: uncomment the sections in main() function to get output of related questions

'''
import numpy as np
import pandas as pd
import datetime as dt
from glob import glob
from sklearn import svm
from sklearn import metrics
from sklearn import ensemble
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare

date_string = "green_tripdata_2015-09"  # Representing september 2015


def get_file_content():

    filename = date_string + ".csv"

    if glob(filename):
        print 'File already downloaded'
        data = pd.read_csv(filename)

        return data

    # Else downloading file
    url = "https://s3.amazonaws.com/nyc-tlc/trip+data/" + filename
    print('Please wait while file is being downloaded')
    data = pd.read_csv(url)
    data.to_csv(url.split('/')[-1])
    return data

# Question 2a and 2b


def plot_histogram(data):
    # histogram of the number of trip distance
    data.Trip_distance.hist(bins=100)
    plt.xlabel('Trip distance')
    plt.ylabel('No. of trips')
    plt.savefig('Question2a.jpg', format='jpg')

    mean_of_data = data['Trip_distance'].mean()
    std_of_data = data['Trip_distance'].std()
    # find points which are within 3 standard deviation
    valid_points = [i for i in data.Trip_distance if abs(
        i - mean_of_data) < 3 * std_of_data]

    plt.hist(valid_points, bins=100)
    plt.title('Data less than 3 standard deviation')
    plt.xlabel('Trip distance')
    plt.ylabel('count')
    plt.savefig('Question2b.jpg', format='jpg')

# Questio 3 a


def stats_by_hour(data):
    data['Pickup_dt'] = data.lpep_pickup_datetime.apply(
        lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    data['Dropoff_dt'] = data.Lpep_dropoff_datetime.apply(
        lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    data['byhour'] = data.Pickup_dt.apply(lambda x: x.hour)
    table1 = data.pivot_table(index='byhour', values='Trip_distance', aggfunc=(
        'mean', 'median')).reset_index()
    table1.columns = ['Hour', 'Mean_distance', 'Median_distance']
    print tabulate(table1.values.tolist(), ["Hour", "Mean distance", "Median distance"])

# Questio 3 b


def airport_area_stats(data):
    airports_trips = data[(data.RateCodeID == 2) | (data.RateCodeID == 3)]
    print "No. of trips to/from airport area: ", airports_trips.shape[0]
    print "Avg. fare airport area: $", airports_trips.Fare_amount.mean()
    print "Avg. total amount airport area: $", airports_trips.Total_amount.mean()
    plt.hist(airports_trips.Trip_distance)
    plt.show()


def predictive_model(data):

    new_data = data.copy()

    # considering only relevant features.
    columns = ["RateCodeID",
               "Pickup_longitude",
               "Pickup_latitude",
               "Dropoff_longitude",
               "Dropoff_latitude",
               "Passenger_count",
               "Trip_distance",
               "Fare_amount",
               "Extra",
               "MTA_tax",
               "Tolls_amount",
               "Total_amount",
               "Payment_type"]

    new_data = new_data[columns]
    new_data = pd.DataFrame(new_data).fillna(0)
    data.Per_TIP = pd.DataFrame(data.Per_TIP).fillna(0)

    # Spliting the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        new_data,
        data.Per_TIP,
        test_size=0.25,
        random_state=33
    )

    # Scaling the data
    scalerX = StandardScaler().fit(X_train)
    scalery = StandardScaler().fit(y_train)
    X_train = scalerX.transform(X_train)
    y_train = scalery.transform(y_train)
    X_test = scalerX.transform(X_test)
    y_test = scalery.transform(y_test)

    # performance measurement function
    def measure_performance(
        X, y, clf,
        show_accuracy=True,
        show_classification_report=True,
        show_confusion_matrix=True,
        show_r2_score=False
    ):
        y_pred = clf.predict(X)
        if show_accuracy:
            print "Accuracy:{0:.3f}".format(
                metrics.accuracy_score(y, y_pred)
            ), "\n"

        if show_classification_report:
            print "Classification report"
            print metrics.classification_report(y, y_pred), "\n"

        if show_confusion_matrix:
            print "Confusion matrix"
            print metrics.confusion_matrix(y, y_pred), "\n"

        if show_r2_score:
            print "Coefficient of determination:{0:.3f}".format(
                metrics.r2_score(y, y_pred)
            ), "\n"

    # training fuction
    def train_and_evaluate(clf, X_train, y_train):
        clf.fit(X_train, y_train)
        print "Coefficient of determination on training set:", clf.score(X_train, y_train)
        # create a k-fold cross validation iterator of k=5 folds
        cv = KFold(X_train.shape[0], 5, shuffle=True,
                   random_state=33)
        scores = cross_val_score(clf, X_train, y_train, cv=cv)
        print "Average coefficient of determination using 5 - fold crossvalidation:", np.mean(scores)

    # trying different classifiers

    # 1. Linear regression with no penalty
    print '\n\n Linear regression\n\n'
    clf_sgd = linear_model.SGDRegressor(
        loss='squared_loss',
        penalty=None,
        random_state=42
    )
    train_and_evaluate(clf_sgd, X_train, y_train)

    # 2. Linear regression with l2 normalization
    print '\n\n Linear regression L2\n\n'
    clf_sgd1 = linear_model.SGDRegressor(
        loss='squared_loss',
        penalty='l2',
        random_state=42
    )
    train_and_evaluate(clf_sgd1, X_train, y_train)

    # 3. SVM linear
    print '\n\n SVM linear\n\n'
    clf_svr = svm.SVR(kernel='linear')
    train_and_evaluate(clf_svr, X_train, y_train)

    # 4. SVM poly
    print '\n\n SVM poly\n\n'
    clf_svr_poly = svm.SVR(kernel='poly')
    train_and_evaluate(clf_svr_poly, X_train, y_train)

    # 5. SVM RBF
    print '\n\n SVM RBF\n\n'
    clf_svr_rbf = svm.SVR(kernel='rbf')
    train_and_evaluate(clf_svr_rbf, X_train, y_train)

    # 6. Tree regressor
    print '\n\n Tree regressor\n\n'
    clf_et = ensemble.ExtraTreesRegressor(
        n_estimators=10,
        compute_importances=True,
        random_state=42
    )
    train_and_evaluate(clf_et, X_train, y_train)

    # printing important features for tree regressor
    print sort(
        zip(clf_et.feature_importances_, boston.feature_names),
        axis=0
    )

    # measure_performance(X_test, y_test, clf_et,
    # 	show_accuracy=False,
    # 	show_classification_report=False,
    #     show_confusion_matrix=True
    #     )


def main():
    # loading the data
    data = get_file_content()

    # Question 1
    # print "Number of rows:", data.shape[0]
    # print "Number of columns: ", data.shape[1]

    # Question 2
    # plot_histogram(data)

    # Question 3
    # stats_by_hour(data)
    # airport_area_stats(data)

    # Question 4 a
    data['Per_TIP'] = 100 * data.Tip_amount / data.Total_amount

    # Question 4 b
    predictive_model(data)

if __name__ == '__main__':
    main()
