# Methodology in *report.ipynb*

# Requirements

Python and Miniconda/Anaconda installed https://docs.conda.io/en/latest/miniconda.html
    
```conda env create -f env.yml```

# CLI to train model

    conda activate $ENV_NAME [default: conda activate juvo_test_envi]
    python main.py

    Usage: main.py [OPTIONS]

    Options:
    -l, --loans TEXT                Path to loans dataset  [default: datasets/Br
                                    azil_DS_loans_2019-11-10_2019-12-05.csv]
    -lp, --loans_prev TEXT          Path to previous loans dataset  [default:
                                    datasets/Brazil_DS_prev_loans.csv]
    -r, --recharges TEXT            Path to recharges dataset  [default: dataset
                                    s/Brazil_DS_recharges_2019-08-10_2019-12-05.
                                    csv]
    -m, --metric TEXT               Metric to evaluate the model. AUC[default] =
                                    roc_auc, Recall = recall, F1=f1,
                                    Accuracy=accuracy  [default: roc_auc]
    -i, --inicial_date TEXT         Inicial date to train the model. Format:
                                    YYYY-MM-DD  [default: 2019-01-01]
    -f, --final_date TEXT           Final date to train the model. Format: YYYY-
                                    MM-DD  [default: 2019-12-31]
    --plot, --no-plot               Show plots. --no-plot to disable  [default:
                                    True]
    --install-completion [bash|zsh|fish|powershell|pwsh]
                                    Install completion for the specified shell.
    --show-completion [bash|zsh|fish|powershell|pwsh]
                                    Show completion for the specified shell, to
                                    copy it or customize the installation.
    --help                          Show this message and exit.
# Datasets
You will find 3 csv files. The files were created from our database as of 2020 Feb 05.

##      *Brazil_DS_loans_2019-11-10_2019-12-05*

It has the loans made for a period of 25 days with following important fields

- Loan_id - unique identifier for a loan
- Uuid - user identifier
- Created_at - time when loan was created
- Paid_at - time when it was paid. If it is missing then loan was not paid as of file creation date
- Amount - amount of loan

A
 loan is considered repaid if it's paid within 60 days.The objective is 
to create a predictive model for loan repayment based on the labels in 
this file.

##       *Brazil_DS_prev_loans*

This has the previous loans taken for users in the above file and should have the same schema.

##       *Brazil_DS_recharges_2019-08-10_2019-12-05*

A user pays for loans by making recharges after taking a loan. This file contains recharges for all users for about 4 months.