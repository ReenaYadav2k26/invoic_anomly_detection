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

    cols = ['paid_before_due', 'credit_gt_debit', 'high_delay', 'amount_mismatch']
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
    data = request.get_json(force=True)

    # STEP 1: Handle Unify wrapper
    if isinstance(data, dict):
        if "body" in data:
            data = data["body"]

    # STEP 2: Convert to list
    if isinstance(data, dict):
        data = [data]

    # STEP 3: Extract "properties" (🔥 MAIN FIX)
    cleaned_data = []
    for item in data:
        if isinstance(item, dict) and "properties" in item:
            cleaned_data.append(item["properties"])
        else:
            cleaned_data.append(item)

    # STEP 4: Validate
    if not isinstance(cleaned_data, list):
        return jsonify({"error": "Invalid input format"}), 400

    print("CLEANED DATA:", cleaned_data)

    df = pd.DataFrame(cleaned_data)

    try:
        # Rename / map fields
        df['amount'] = df['total_due']
        df['usage_units'] = 0
        df['usage_rate'] = 0
        df['paid_date'] = df['posting_date']

        # Convert numeric fields
        df['debit'] = df['debit'].astype(float)
        df['credit'] = df['credit'].astype(float)
        df['amount'] = df['amount'].astype(float)

        # Convert dates
        for col in ['due_date', 'paid_date', 'bill_from_date', 'bill_thru_date']:
            df[col] = pd.to_datetime(df[col])

        df = calculate_anomaly(df)

        return jsonify(
            df[['invoice_no', 'anomaly_score', 'is_anomaly', 'trigger_reasons']]
            .to_dict(orient="records")
        )

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
