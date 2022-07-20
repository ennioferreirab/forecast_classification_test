import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List
from .feature_engineering import FeatureEngineering
from .fit_model import FitModel

@dataclass
class ForecastDefault:
    '''
    docstring
    '''
    loans_hist: pd.DataFrame
    recharges_hist: pd.DataFrame
    estimators_list: List
    eval_metric: str = 'roc_auc'
    inicial_date: str = '2000-01-01'
    limit_date: str = '2019-12-05'
    days_to_default: int = 60
    fill_na:bool = True

    def __post_init__(self):
        date_format = '%Y-%m-%d'
        self.inicial_date = datetime.strptime(self.inicial_date,date_format)
        self.limit_date = datetime.strptime(self.limit_date,date_format)
        self.loans_hist['paid_at'] = self.loans_hist['paid_at'].fillna(self.limit_date + timedelta(days=1))
        last_date = self.limit_date - timedelta(days=self.days_to_default)
        

        self.loans = self.loans_hist[self.loans_hist['created_at'] > self.inicial_date].copy()
        self.recharges = self.recharges_hist[self.recharges_hist['recharge_timestamp'] > self.inicial_date].copy()
        

        train_loans = self.loans[self.loans['created_at'] < last_date]
        test_loans = self.loans[((self.loans['created_at'] > last_date) & (self.loans['created_at'] < self.limit_date))]
        train_recharges = self.recharges[self.recharges['recharge_timestamp'] < last_date]
        
        print('last_date',last_date)
        print('limit_date',test_loans['created_at'].max())

        
        print('train_loans: ',train_loans.shape)
        print('test_loans: ',test_loans.shape)


        fe = FeatureEngineering(train_loans,test_loans,train_recharges,days_to_default=self.days_to_default)

        train_df = fe.train_loans.merge(fe.train_recharges, on='uuid', how='left')
        train_df = fe.remove_perfect_correlation(train_df)

        if self.fill_na:
            train_df.fillna(0,inplace=True)

        self.model = FitModel(train_df,fe.test_loans[['uuid','target']],self.estimators_list, self.eval_metric)