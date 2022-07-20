import joblib
import pandas as pd

def test_load_dfs(df_1,df_2):
    template_1 = joblib.load('tests/loans_template.pkl').dtypes
    template_3 = joblib.load('tests/recharges_template.pkl').dtypes
    assert (df_1.dtypes == template_1).all()
    assert (df_2.dtypes == template_3).all()

    return df_1,df_2


def load_dfs(paths=[
    'datasets/Brazil_DS_loans_2019-11-10_2019-12-05.csv',
    'datasets/Brazil_DS_prev_loans.csv',
    'datasets/Brazil_DS_recharges_2019-08-10_2019-12-05.csv']):
    
    loans_actual = pd.read_csv(paths[0],date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'),parse_dates=['created_at','paid_at'])
    loans_prev = pd.read_csv(paths[1],date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'),parse_dates=['created_at','paid_at'])
    loans = pd.concat([loans_actual,loans_prev],ignore_index=True)
    recharges = pd.read_csv(paths[2],date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'),parse_dates=['recharge_timestamp'])
    #return  test_load_dfs(loans, recharges)
    return  loans, recharges


loans, recharges = load_dfs()