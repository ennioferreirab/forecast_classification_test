import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List
from .feature_engineering import FeatureEngineering
from .fit_model import FitModel

import logging
        
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
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
        self.loans_hist['paid_at'] = self.loans_hist['paid_at'].fillna(self.limit_date + timedelta(days=360))
        
        last_date = self.loans_hist[self.loans_hist['created_at'] < self.limit_date]['created_at'].max() \
            - timedelta(days=self.days_to_default)
        

        self.loans = self.loans_hist[self.loans_hist['created_at'] > self.inicial_date].copy()
        self.recharges = self.recharges_hist[self.recharges_hist['recharge_timestamp'] > self.inicial_date].copy()
        

        self.train_loans = self.loans[self.loans['created_at'] < last_date]
        self.test_loans = self.loans[((self.loans['created_at'] > last_date) & (self.loans['created_at'] < self.limit_date))]
        train_recharges = self.recharges[self.recharges['recharge_timestamp'] < last_date]
        
        #switch print function to logging
        
        logger.debug(f"first_date : {self.train_loans['created_at'].min()}")
        logger.debug(f"last_date {last_date}")
        logger.debug(f"limit_date {self.test_loans['created_at'].max()}")

        
        logger.debug(f"train_loans: {self.train_loans.shape}")
        logger.debug(f"test_loans: {self.test_loans.shape}")


        fe = FeatureEngineering(self.train_loans,self.test_loans,train_recharges,days_to_default=self.days_to_default)

        self.train_df = fe.train_loans.merge(fe.train_recharges, on='uuid', how='left')
        self.corr_df = self.train_df.copy()
        self.train_df = fe.remove_perfect_correlation(self.train_df)

        if self.fill_na:
            self.train_df.fillna(0,inplace=True)

        self.model = FitModel(self.train_df,fe.test_loans[['uuid','target']],self.estimators_list, self.eval_metric)