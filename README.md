# Methodology in *report.ipynb*

# Requirements

Python and Miniconda/Anaconda installed https://docs.conda.io/en/latest/miniconda.html
    
```conda env create -f env.yml```

# CLI to train model

    conda activate $ENV_NAME [default: conda activate juvo_test_envi]
    python main.py help
    python main.py
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