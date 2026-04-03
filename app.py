from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

def calculate_anomaly(df):
    df['payment_delay'] = (df['paid_date'] - df['due_date']).dt.days
    df['bill_days'] = (df['bill_thru_date'] - df['bill_from_date']).dt.days
    df['amount_diff'] = df['debit'] - df['amount']
    df['usage_charge'] = df['usage_units'] * df['usage_rate']

    df['paid_before_due'] = df['paid_date'] < df['due_date']
    df['credit_gt_debit'] = df['credit'] > df['debit']
    df['high_delay'] = df['payment_delay'] > 30
    df['amount_mismatch'] = abs(df['amount_diff']) > 20

    cols = ['paid_before_due','credit_gt_debit','high_delay','amount_mismatch']
    df['anomaly_score'] = df[cols].sum(axis=1)
    df['is_anomaly'] = df['anomaly_score'] >= 2

    df['trigger_reasons'] = df[cols].apply(
        lambda x: [col for col in cols if x[col]], axis=1
    )

    return df

@app.route("/")
def home():
    return "Anomaly Detection API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)

    for col in ['due_date','paid_date','bill_from_date','bill_thru_date']:
        df[col] = pd.to_datetime(df[col])

    df = calculate_anomaly(df)

    return jsonify(df[['invoice_no','anomaly_score','is_anomaly','trigger_reasons']].to_dict(orient="records"))
