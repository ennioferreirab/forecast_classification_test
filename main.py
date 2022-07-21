from typing import Optional
from functions.read import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from functions.feature_default import ForecastDefault
import logging
import warnings
import joblib
warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import typer




'''def _version_callback(value: bool) -> None:

    if value:

        typer.echo(f"{__app_name__} v{__version__}")

        raise typer.Exit()'''

def main(

    path_loans: Optional[str] = typer.Option(

        'datasets/Brazil_DS_loans_2019-11-10_2019-12-05.csv',

        "--loans",

        "-l",

        help="Path to loans dataset",

        # callback=_version_callback,

        #is_eager=True
        ),
    path_loans_prev: Optional[str] = typer.Option(

        'datasets/Brazil_DS_prev_loans.csv',

        "--loans_prev",

        "-lp",

        help="Path to previous loans dataset"
        ),
    path_recharges: Optional[str] = typer.Option(

        'datasets/Brazil_DS_recharges_2019-08-10_2019-12-05.csv',

        "--recharges",

        "-r",

        help="Path to recharges dataset"
        ),
    eval_metric: str=typer.Option(
        'roc_auc',
        "--metric",
        "-m",
        help="Metric to evaluate the model. AUC[default] = roc_auc, Recall = recall, F1=f1, Accuracy=accuracy"
    ),
    inicial_date:Optional[str]=typer.Option(
        '2019-01-01',
        "--inicial_date",
        "-i",
        help="Inicial date to train the model. Format: YYYY-MM-DD"
    ),
    limit_date:Optional[str]=typer.Option(
        '2019-12-31',
        "--final_date",
        "-f",
        help="Final date to train the model. Format: YYYY-MM-DD"
    ),
    plot:Optional[bool]=typer.Option(
        True,
        "--plot",
        "--no-plot",
        help="Show plots. --no-plot to disable"
    ),
    ) -> None:
    if plot == '--plot':
        plot = True
    elif plot == '--no-plot':
        plot = False
        
    loans, recharges = load_dfs(
        path_loans,
        path_loans_prev,
        path_recharges
    )

    forecast = ForecastDefault(
    loans_hist=loans, 
    recharges_hist=recharges, estimators_list=[
                                                LogisticRegression,
                                                XGBClassifier,
                                                RandomForestClassifier,
                                                ],
    eval_metric=eval_metric,
    inicial_date=inicial_date,days_to_default=60,limit_date=limit_date,plot=plot
    )
    joblib.dump(forecast.model.est,'model.pkl')
    return

if __name__ == "__main__":
    typer.run(main)