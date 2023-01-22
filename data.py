import pandas as pd
import sklearn
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt  # for graphics
import seaborn as sns  # for nicer graphics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def variables_treatment(bank_aditional):
    '''
        VARIABLES TREATMENT
        Categorize by labeled encoded and Normalize variables for standard values and knn usage
    '''

    '''
            About the client
    '''

    bank_client = bank_aditional.iloc[: , 0:8]
    bank_client.keys()

    #Label class for all categorical related to person by labeled encoded tool
    labelencoder_X = LabelEncoder()
    bank_client['job'] = labelencoder_X.fit_transform(bank_client['job'])
    bank_client['marital'] = labelencoder_X.fit_transform(bank_client['marital'])
    bank_client['education'] = labelencoder_X.fit_transform(bank_client['education'])
    bank_client['default'] = labelencoder_X.fit_transform(bank_client['default'])
    bank_client['housing'] = labelencoder_X.fit_transform(bank_client['housing'])
    bank_client['loan'] = labelencoder_X.fit_transform(bank_client['loan'])

    # function to creat group of ages as cathegorical tag. Percentile helps to define 4 groups, the last one where outliers
    bank_client.loc[bank_client['age'] <= 32, 'age'] = 1
    bank_client.loc[(bank_client['age'] > 32) & (bank_client['age'] <= 47), 'age'] = 2
    bank_client.loc[(bank_client['age'] > 47) & (bank_client['age'] <= 74), 'age'] = 3

    '''
                About the campaing data: categorical by labeled enconded treatement
    '''
    bank_campaing_related = bank_aditional.iloc[:, 8:11]
    bank_campaing_related['contact'] = labelencoder_X.fit_transform(bank_campaing_related['contact'])
    bank_campaing_related['month'] = labelencoder_X.fit_transform(bank_campaing_related['month'])
    bank_campaing_related['day'] = labelencoder_X.fit_transform(bank_campaing_related['day'])


    bank_extra = bank_aditional.loc[:, ['campaign', 'pdays', 'previous', 'poutcome']]
    bank_extra['poutcome'].replace(['unknown', 'failure', 'success','other'], [1, 2, 3,0], inplace=True)

    return bank_client, bank_campaing_related, bank_extra


def model(bank_final, y):
    bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                             'contact', 'month', 'day','campaign', 'previous', 'poutcome']]
    '''
        KNN set
    '''

    X_trainK, X_testK, y_trainK, y_testK = train_test_split(bank_final, y, test_size=0.25, random_state=101)

    # Neighbors
    neighbors = np.arange(0, 25)

    # Create empty list that will hold cv scores
    cv_scores = []

    # Perform 15-fold cross validation:
    for k in neighbors:
        k_value = k + 1
        knn = KNeighborsClassifier(n_neighbors=k_value, weights='uniform', p=2, metric='euclidean')
        kfold = model_selection.KFold(n_splits=15, random_state=123)
        scores = model_selection.cross_val_score(knn, X_trainK, y_trainK, cv=kfold, scoring='accuracy')
        cv_scores.append(scores.mean() * 100)
        logger.info(f'''({k_value}, {scores.mean() * 100}, {scores.std() * 100})''')

    optimal_k = neighbors[cv_scores.index(max(cv_scores))]
    logger.info(f"N number of neighbors choise {optimal_k} by cross valiadation highest score {cv_scores[optimal_k]}")

    plt.plot(neighbors, cv_scores)
    plt.xlabel('o of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()


    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size=0.25, random_state=101)
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    knn.fit(X_train, y_train)
    knnpred = knn.predict(X_test)

    '''
        Summary    
    '''

    logger.info(f'''{confusion_matrix(y_test, knnpred)}''')
    logger.info(f'''Accuracy by area under ROC {round(accuracy_score(y_test, knnpred), 2) * 100}''')
    KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())


    '''
        ROC
    '''
    probs = knn.predict_proba(X_test)
    preds = probs[:, 1]
    try:
        falsepositiverates_knn, truepositoverates_knn, thresholdknn = metrics.roc_curve(y_test, preds)
        roc_aucknn = metrics.auc(falsepositiverates_knn, truepositoverates_knn)

        ax_arr[0, 2].plot(falsepositiverates_knn, truepositoverates_knn, 'b', label='AUC = %0.2f' % roc_aucknn)
        ax_arr[0, 2].plot([0, 1], [0, 1], 'r--')
        ax_arr[0, 2].set_title('Receiver Operating Characteristic KNN ', fontsize=20)
        ax_arr[0, 2].set_ylabel('True Positive Rate', fontsize=20)
        ax_arr[0, 2].set_xlabel('False Positive Rate', fontsize=15)
        ax_arr[0, 2].legend(loc='lower right', prop={'size': 16})
    except Exception as e:
        logger.error(f'''{e}''')


def main():
    '''
            SOURCE bank-full
    '''
    bank_full = pd.read_csv('bank-full.csv', sep=';')
    bank_full.info()
    #cleaning outliers in balance and age dimensions
    bank_full = bank_full[bank_full['age'] < 74]
    bank_full = bank_full[bank_full['balance'] < 40060]

    # Building dependent variable as standar cat by dummy
    y = pd.get_dummies(bank_full['y'], columns=['y'], prefix=['y'], drop_first=True)

    # Building independent variables to class in variables treatment function
    bank_client, bank_related, bank_extra = variables_treatment(bank_full)
    bank_final= pd.concat([bank_client, bank_related, bank_extra], axis = 1)
    logger.info(f'''DF keys {bank_final.keys()}''')
    
    #KNN choise for first classification approach
    model(bank_final, y)
