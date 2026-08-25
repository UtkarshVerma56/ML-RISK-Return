import streamlit as sttt
import requestssss
import json
import plotly.graph_objects as go

st.set_page_config(page_title="Return Risk Scorer", page_icon="📦", layout="wide")

API_URL = "https://ml-risk-return.onrender.com/predict"

st.title("📦 Return Risk Scorer")
st.caption(
    "Predicts return risk for e-commerce orders — XGBoost model with SHAP explainability, "
    "trained on 18,000 synthetic orders, served via FastAPI on Render."
)

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Order Details")

    category = st.selectbox("Category", ["apparel", "footwear", "electronics", "home_goods", "beauty", "books"])
    payment_method = st.selectbox("Payment Method", ["cod", "UPI", "credit_card", "debit_card", "netbanking"])
    order_amount = st.number_input("Order Amount (₹)", min_value=100.0, max_value=50000.0, value=1200.0, step=100.0)
    discount_pct = st.slider("Discount %", 0, 50, 20)
    customer_past_orders = st.number_input("Customer's Past Orders", min_value=0, max_value=50, value=2)
    customer_past_returns = st.number_input(
        "Customer's Past Returns", min_value=0,
        max_value=int(customer_past_orders) if customer_past_orders > 0 else 0, value=0
    )
    account_age_days = st.number_input("Account Age (days)", min_value=1, max_value=3000, value=180)
    day_of_week = st.selectbox("Day of Week", options=list(range(7)),
                                 format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    hour_of_day = st.slider("Hour of Day", 0, 23, 14)
    address_mismatch = st.checkbox("Billing/Shipping Address Mismatch")

    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction Result")

    if predict_btn:
        payload = {
            "category": category,
            "payment_method": payment_method,
            "order_amount": order_amount,
            "discount_pct": float(discount_pct),
            "customer_past_orders": int(customer_past_orders),
            "customer_past_returns": int(customer_past_returns),
            "account_age_days": int(account_age_days),
            "day_of_week": int(day_of_week),
            "hour_of_day": int(hour_of_day),
            "address_mismatch": 1 if address_mismatch else 0
        }

        with st.spinner("Scoring order... (first request may take ~30-50s if the server was idle)"):
            try:
                response = requests.post(API_URL, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()

                risk_score = result['risk_score']
                risk_flag = result['risk_flag']
                threshold = result['threshold_used']
                top_factors = result['top_factors']

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    number={'suffix': "%"},
                    title={'text': "Return Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#e74c3c" if risk_flag else "#2ecc71"},
                        'steps': [
                            {'range': [0, threshold*100], 'color': "#d5f4e6"},
                            {'range': [threshold*100, 100], 'color': "#fadbd8"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

                if risk_flag:
                    st.error(f"⚠️ **HIGH RISK** — flagged above threshold ({threshold:.2f})")
                else:
                    st.success(f"✅ **LOW RISK** — below threshold ({threshold:.2f})")

                st.markdown("**Top contributing factors:**")
                for factor in top_factors:
                    st.write(f"- {factor}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API: {e}")
    else:
        st.info("Fill in order details and click **Predict Risk** to see the result.")

st.divider()

st.subheader("Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("PR-AUC", "0.331", help="~1.9x better than random baseline (~0.17)")
m2.metric("ROC-AUC", "0.706")
m3.metric("Cost-Optimal Threshold", "0.41")
m4.metric("Cost Savings vs Default", "3.7%", delta="₹3,500 saved")

st.caption(
    "Trained on 18,000 synthetic orders. Full methodology, model card, and source: "
    "[GitHub repo](https://github.com/UtkarshVerma56/ML-RISK-Return)"
)
