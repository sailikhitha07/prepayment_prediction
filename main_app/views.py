from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from joblib import load
import json

file_path = './savedModels/pipelineOfModels.joblib'
pipeline = load(file_path)

with open('./savedModels/frequency_encoding.json', 'r') as f:
    frequency_encoding = json.load(f)

def derive_features(X):

    X = X.copy()

    bins_dti = [0, 20, 30, 40, 50, 100]
    labels_dti = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']

    p = X['OrigUPB']
    r = X['OrigInterestRate'] / (100 * 12)
    n = X['OrigLoanTerm']
    X['MonthlyPayment'] = round(p * r * ((1 + r) ** n / ((1 + r) ** n - 1)), 2)

    P = X['OrigUPB']
    r = X['OrigInterestRate'] / (100 * 12)
    E = X['MonthlyPayment']
    M = X['MonthsInRepayment']
    X['CurrentPrincipal'] = round((P * (1 + r) ** M) - (E * (((1 + r) ** M - 1) / r)), 2)

    X['total_amount'] = X['MonthlyPayment'] * X['OrigLoanTerm']
    X['interest_amount'] = X['total_amount'] - X['OrigUPB']

    X['MonthlyIncome'] = round(X['MonthlyPayment'] / (X['DTI'] / 100), 2)

    X['DTIBin'] = pd.cut(X['DTI'], bins=bins_dti, labels=labels_dti, right=False)
    X['DTIBins'] = X['DTIBin'].map({'Low': 0, 'Moderate': 1, 'High': 2, 'Very High': 3, 'Extreme': 4})

    X = X.drop(['DTIBin', 'DTI'], axis=1)

    return X

def formInfo(request):
    if request.method == 'POST':
        try:
            channel = request.POST.get('Channel')
            loan_purpose = request.POST.get('LoanPurpose')
            seller_name = request.POST.get('SellerName')
            servicer_name = request.POST.get('ServicerName')
            mip = float(request.POST.get('MIP'))
            units = int(request.POST.get('Units'))
            dti = float(request.POST.get('DTI'))
            orig_upb = float(request.POST.get('OrigUPB'))
            orig_interest_rate = float(request.POST.get('OrigInterestRate'))
            orig_loan_term = int(request.POST.get('OrigLoanTerm'))
            num_borrowers = int(request.POST.get('NumBorrowers'))
            months_delinquent = int(request.POST.get('MonthsDelinquent'))
            months_in_repayment = int(request.POST.get('MonthsInRepayment'))
            credit_range_encoded = int(request.POST.get('CreditRange_Encoded'))
            ltv_range_encoded = int(request.POST.get('LTVRange_Encoded'))
            repay_range_encoded = int(request.POST.get('Repay_range_Encoded'))
            first_time_homebuyer_encoded = int(request.POST.get('FirstTimeHomebuyer_Encoded'))

            # Create input data dictionary
            input_data = {
                'MIP': [mip],
                'Units': [units],
                'DTI': [dti],
                'OrigUPB': [orig_upb],
                'OrigInterestRate': [orig_interest_rate],
                'Channel': [channel],
                'LoanPurpose': [loan_purpose],
                'OrigLoanTerm': [orig_loan_term],
                'NumBorrowers': [num_borrowers],
                'SellerName': [seller_name],
                'ServicerName': [servicer_name],
                'MonthsDelinquent': [months_delinquent],
                'MonthsInRepayment': [months_in_repayment],
                'CreditRange_Encoded': [credit_range_encoded],
                'LTVRange_Encoded': [ltv_range_encoded],
                'Repay_range_Encoded': [repay_range_encoded],
                'FirstTimeHomebuyer_Encoded': [first_time_homebuyer_encoded]
            }

            # Process the input data
            df_input = pd.DataFrame(input_data)
            df_input = apply_frequency_encoding(df_input, frequency_encoding)
            df_input = derive_features(df_input)

            desired_order = [
                'MIP', 'Units', 'OrigUPB', 'OrigInterestRate', 'Channel',
                'LoanPurpose', 'OrigLoanTerm', 'NumBorrowers', 'SellerName',
                'ServicerName', 'MonthsDelinquent', 'MonthsInRepayment',
                'DTIBins', 'CreditRange_Encoded', 'LTVRange_Encoded',
                'Repay_range_Encoded', 'FirstTimeHomebuyer_Encoded',
                'MonthlyPayment', 'CurrentPrincipal', 'total_amount',
                'interest_amount', 'MonthlyIncome'
            ]
            
            # Reorder the DataFrame columns
            df_input = df_input[desired_order]

            if not all(col in df_input.columns for col in desired_order):
                missing_cols = [col for col in desired_order if col not in df_input.columns]
                return render(request, 'result.html', {'error': f"Missing columns in input data: {', '.join(missing_cols)}"})
            
            X_new = df_input

            # Make predictions
            class_prediction, regression_prediction = pipeline.predict(X_new)
            
            print('classification_prediction:', int(class_prediction[0]))
            print('regression_prediction:', regression_prediction[0])
            
            context = {
                'classification_prediction': int(class_prediction[0]),
                'regression_prediction': regression_prediction[0] if not np.isnan(regression_prediction[0]) else 'N/A'
            }

            return render(request, 'result.html', context)
        except Exception as e:
            return render(request, 'result.html', {'error': f"An error occurred: {e}"})
    else:
        return render(request, 'main.html')

def predictor(request):
    return render(request, 'main.html')

def apply_frequency_encoding(df, encoding_dict):
    for col, freq_encoding in encoding_dict.items():
        if col in df.columns:
            df[col] = df[col].map(freq_encoding).fillna(0)
    return df
