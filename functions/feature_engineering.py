import pandas as pd
from dataclasses import dataclass


@dataclass
class FeatureEngineering:
    train_loans: pd.DataFrame
    test_loans: pd.DataFrame
    train_recharges: pd.DataFrame
    days_to_default:int = 60
    

    def __post_init__(self):

        self.train_loans = self.__feature_eng_loans(self.train_loans.copy())
        self.test_loans = self.__feature_eng_loans(self.test_loans.copy())
        self.train_recharges = self.__feature_eng_recharges(self.train_recharges.copy())
        

    def __feature_eng_loans(self,df_l: pd.DataFrame) -> pd.DataFrame:
            '''
            Feature engineering for loans dataframe using historical data
            already_default: 0 if not defaulted, 1 if defaulted
            median_amount_loan: median amount of all previous loans paid
            count_loans: number of previous loans paid

            :param df_l: loans dataframe
            :return: feature engineered loans dataframe
            '''
            df_l['paid_days_interval'] = (df_l['paid_at'] - df_l['created_at'])
            df_l['target'] = df_l['paid_days_interval'].apply(lambda x: 1 if x.days > self.days_to_default else 0)

            already_default = df_l.groupby('uuid').sum()['target'].apply(lambda x: 1 if x > 0 else 0)
            median_amount_loan = df_l.query('target == 0').groupby('uuid').median()['amount']
            count_loans = df_l.query('target == 0').groupby('uuid').count()['amount']

            out_df = pd.DataFrame(df_l.groupby('uuid').count().index)
            out_df = out_df.join(already_default,on='uuid',how='left')
            out_df = out_df.join(median_amount_loan,on='uuid',how='left',rsuffix='_sum')
            out_df = out_df.join(count_loans,on='uuid',how='left',rsuffix='_count')
            out_df.columns = ['uuid','target','median_amount_loan','count_loans']
            return out_df
        
    def __feature_eng_recharges(self,df_r: pd.DataFrame) -> pd.DataFrame:
        '''
        Feature engineering for recharges dataframe using historical data
        freq_recharges_weekly: mean frequency recharges per week
        recharges_weekly: median frequency recharges per week
        delta_after_recharge: difference between balance after recharge and recharge value

        :param df_r: recharges dataframe
        :return: feature engineered recharges dataframe
        '''
        df_r['delta_after_recharge'] = df_r['balance_after_recharge'] - df_r['recharge_value']
        df_r['back_recharge_timestamp'] = pd.to_datetime(df_r['recharge_timestamp']) - pd.to_timedelta(7, unit='d')
        max_date = df_r['back_recharge_timestamp'].max()
        min_date = df_r['back_recharge_timestamp'].min()
        count_weeks = (max_date - min_date).days // 7
        weekly_df = df_r \
            .groupby(['uuid', pd.Grouper(key='recharge_timestamp', freq='W-MON')]) \
            .count() \
            .groupby('uuid')
        
        freq_recharges_weekly = weekly_df.sum()['recharge_value']/count_weeks
        recharges_weekly = weekly_df.median()['recharge_value']/count_weeks
        delta_after_recharges = df_r.groupby('uuid').median()['delta_after_recharge']
        
        out_df = pd.DataFrame(df_r.groupby('uuid').count().index)
        out_df = out_df.join(freq_recharges_weekly,on='uuid',how='left',rsuffix='_median')
        out_df = out_df.join(recharges_weekly,on='uuid',how='left',rsuffix='_median')
        out_df = out_df.join(delta_after_recharges,on='uuid',how='left')
        out_df.columns = ['uuid','freq_recharges_weekly','recharges_weekly','delta_after_recharges']
        return out_df

    @staticmethod
    def remove_perfect_correlation(df: pd.DataFrame,max_cor = 0.95) -> pd.DataFrame:
        ''''
        Remove columns with perfect correlation with other columns
        :param df: dataframe
        :return: dataframe without columns with perfect correlation
        
        corr_matrix = df.corr()
        corr_matrix.loc[:, :] = np.tril(corr_matrix.values, k=-1)
        cols_to_drop = corr_matrix.loc[:,(corr_matrix.abs() > max_cor).any()]
        print((corr_matrix.abs() > max_cor).any())
        df = df.drop(cols_to_drop,axis=1)
        TODO
        '''
        df = df.drop('count_loans',axis=1)
        return df