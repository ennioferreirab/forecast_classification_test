import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, \
    recall_score, accuracy_score, f1_score, plot_confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import logging
logger = logging.getLogger()
@dataclass
class FitModel:
    '''
    Class to fit the model
    :param train_df: dataframe with training data
    :param fe_test_df: dataframe containing forward targets to evaluate performance metrics
    :param estimator_list: list of estimators to use in the model
    '''
    train_df: pd.DataFrame
    fe_test_df: pd.DataFrame
    estimators_list: List
    eval_metric: str
    plot:bool = True


    def __post_init__(self):
        
        
        self.train_X, self.train_y = self.__under_sampling(self.train_df.copy())

        self.test_df = self.train_X.merge(self.fe_test_df,on='uuid',how='inner')

        self.train_X.drop('uuid',axis=1,inplace=True)
        self.test_df.drop('uuid',axis=1,inplace=True)

        logger.debug(f'Dist train_y target : \n {self.train_y.value_counts()}')
        self.test_X = self.test_df.drop('target',axis=1)
        self.test_y = self.test_df['target']
        logger.debug(f'Dist test_y target : \n {self.test_y.value_counts()}')

        logger.debug(f'train_X:  {self.train_X.shape}')
        logger.debug(f'train_y:  {self.train_y.shape}')
        logger.debug(f'test_X:  {self.test_X.shape}')
        logger.debug(f'test_y:  {self.test_y.shape}')
                

        #Vai mudar para aceitar varios estimadores mas agora vai sobreescrever o modelo

        self.store_model_results = {}
        for est in self.estimators_list:
            self.__kfold_cross_validation(est,k=5)

        self.__best_model()
        logger.info(f'Best model :{self.est.__name__}')
        self.__fit(self.est,X=self.train_X,y=self.train_y)
        eval_metric = self.evaluate_model(self.test_X,self.test_y)
        logger.info(f'Test sampling {self.eval_metric} : {round(eval_metric,3)}')

        self.__make_table_plot_reports()
    
    def __plot_roc_curve(self):
        y_score = self.est.predict_proba(self.test_X)[:,1]

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(self.test_y, y_score)
        roc_auc = auc(fpr, tpr)

        # Compute micro-average ROC curve and ROC area
        plt.figure()
        lw = 2
        plt.plot(
        fpr, tpr, color='darkorange',
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()

    def __make_table_plot_reports(self):
        #print classification reports and confusion matrix
        logger.info(f'Classification report \n {classification_report(self.test_y, self.predict(self.test_X))}')
        logger.info(f'Confusion matrix \n {confusion_matrix(self.test_y, self.predict(self.test_X))}')
        if self.plot:
            plot_confusion_matrix(self.est, self.test_X, self.test_y,cmap='Blues')  
            plt.show()
            self.__plot_roc_curve()

    def __kfold_cross_validation(self,estimator,k=5):
            results_metrics = []
            kf = KFold(n_splits=5,shuffle=True,random_state=42)
            for train_index, test_index in kf.split(X=self.train_X):
                k_X, k_test_X = self.train_X.iloc[train_index], self.train_X.iloc[test_index]
                k_y, k_test_y = self.train_y.iloc[train_index], self.train_y.iloc[test_index]
                self.__fit(estimator,X=k_X,y=k_y)
                results_metrics.append(round(self.evaluate_model(k_test_X,k_test_y),3))
            self.store_model_results[estimator.__name__] = results_metrics
            logger.info(f'{estimator} : Training sample metrics | {results_metrics} | mean: {round(np.mean(results_metrics),3)}')

    def __best_model(self):
        '''
        docstring
        '''
        mean_results = [np.mean(metrics) for metrics in self.store_model_results.values()]
        best_model = self.estimators_list[np.argmax(mean_results)]
        self.est = best_model

    def evaluate_model(self,test_X,test_y):
        if self.eval_metric == 'f1':
            return f1_score(test_y, self.predict(test_X))
        if self.eval_metric == 'acc':
            return accuracy_score(test_y, self.predict(test_X))
        if self.eval_metric == 'recall':
            return recall_score(test_y, self.predict(test_X))
        if self.eval_metric == 'roc_auc':
            return roc_auc_score(test_y, self.predict(test_X))
        
        raise(ValueError('Evaluation metric not supported. Try: f1, acc, recall, roc_auc'))

    
    def predict(self,X: pd.DataFrame) -> pd.Series:
        '''
        Predict using the fitted model
        :param X: dataframe with features
        :return: predictions
        '''
        #pred_df = pd.DataFrame(X.index)
        #pred_df['default_probability'] = self.est.predict(X)
        pred_df = self.est.predict(X)
        return pred_df


    def __fit(self,estimator,X,y):
        '''
        docstring
        '''
        self.est = estimator()
        self.est.fit(X, y)

    
    def __under_sampling(self,df: pd.DataFrame) -> pd.DataFrame:
        '''
        Under sampling of dataframe
        :param df: dataframe
        :return: undersampled dataframe
        '''
        rus = RandomUnderSampler(random_state=42)
        train_X, train_y = rus.fit_resample(
            self.train_df.drop(['target'],axis=1),self.train_df['target'])
        return train_X, train_y