import streamlit as st
import pandas as pd
import requests
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Finlora Fraud Detection",
    page_icon="💳",
    layout="wide"
)

CSV_PATH = (
    "/Users/Dell/Frudulent-transaction-detection-for-Finlora-Company/"
    "Finlora_Dataset/Artifacts/"
    "cleaned_customer_transaction_data.csv"
)

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    return df


try:
    df = load_data()

except Exception as e:
    st.error(f"Unable to load historical transaction data: {e}")
    st.stop()


# ============================================================
# HELPER FUNCTION
# ============================================================

def unique_values(column):
    """
    Get unique non-null values from the historical dataset.
    """
    if column not in df.columns:
        return []

    return sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# ============================================================
# TITLE
# ============================================================

st.title("💳 Finlora Fraud Detection Dashboard")

st.markdown(
    """
    ### Transaction Risk Assessment

    Select an existing customer and enter the details of a new
    transaction. Customer and account information are preloaded
    from the historical Finlora transaction dataset.

    You can edit the preloaded account information if the new
    transaction contains updated information.
    """
)

st.divider()


# ============================================================
# CUSTOMER SELECTION
# ============================================================

st.subheader("👤 Customer Information")

if "customer_id" not in df.columns:
    st.error(
        "The historical dataset does not contain a customer_id column."
    )
    st.stop()


customer_ids = sorted(
    df["customer_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ------------------------------------------------------------
# Customer selector
# ------------------------------------------------------------

selected_customer = st.selectbox(
    "Select Customer",
    customer_ids,
    key="selected_customer"
)


# ============================================================
# GET CUSTOMER'S HISTORICAL INFORMATION
# ============================================================

customer_data = df[
    df["customer_id"].astype(str) == selected_customer
].copy()


if customer_data.empty:
    st.warning("No historical information found for this customer.")
    st.stop()


# ------------------------------------------------------------
# Use the customer's most recent transaction
# ------------------------------------------------------------

if "timestamp" in customer_data.columns:

    customer_data["timestamp"] = pd.to_datetime(
        customer_data["timestamp"],
        errors="coerce"
    )

    customer_record = customer_data.sort_values(
        "timestamp"
    ).iloc[-1]

else:

    customer_record = customer_data.iloc[-1]


# ============================================================
# UPDATE PRELOADED ACCOUNT INFORMATION
# WHEN CUSTOMER CHANGES
# ============================================================

if st.session_state.get("loaded_customer") != selected_customer:

    st.session_state["loaded_customer"] = selected_customer

    st.session_state["home_country"] = str(
        customer_record.get("home_country", "")
    )

    st.session_state["kyc_tier"] = str(
        customer_record.get("kyc_tier", "")
    )

    st.session_state["account_age_days"] = int(
        customer_record.get("account_age_days", 0)
    )

    st.session_state["device_trust_score"] = float(
        customer_record.get("device_trust_score", 0)
    )

    st.session_state["risk_score_internal"] = float(
        customer_record.get("risk_score_internal", 0)
    )

    st.session_state["corridor_risk"] = float(
        customer_record.get("corridor_risk", 0)
    )

    st.session_state["ip_risk_score"] = float(
        customer_record.get("ip_risk_score", 0)
    )


# ============================================================
# PRELOADED ACCOUNT INFORMATION
# ============================================================

st.markdown("#### Preloaded Account Information")

st.caption(
    "These values are automatically loaded from the customer's "
    "most recent historical transaction. You can click and edit them."
)


# ------------------------------------------------------------
# Row 1
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.text_input(
        "Customer ID",
        value=str(selected_customer),
        disabled=True,
        key="display_customer_id"
    )


with col2:

    home_country_options = unique_values("home_country")

    if (
        st.session_state["home_country"]
        not in home_country_options
        and st.session_state["home_country"] != ""
    ):
        home_country_options = [
            st.session_state["home_country"]
        ] + home_country_options

    home_country = st.selectbox(
        "Home Country",
        home_country_options,
        index=home_country_options.index(
            st.session_state["home_country"]
        )
        if st.session_state["home_country"] in home_country_options
        else 0,
        key="home_country"
    )


with col3:

        kyc_options = unique_values("kyc_tier")

        if (
            st.session_state["kyc_tier"]
            not in kyc_options
            and st.session_state["kyc_tier"] != ""
    ):
            kyc_options = [
            st.session_state["kyc_tier"]
        ] + kyc_options

    # Cleaned selectbox without the conflicting index parameter
        kyc_tier = st.selectbox(
        "KYC Tier",
        kyc_options,
        key="kyc_tier"
    )


with col4:

    account_age_days = st.number_input(
        "Account Age (days)",
        min_value=0,
        value=st.session_state["account_age_days"],
        step=1,
        key="account_age_days"
    )


# ------------------------------------------------------------
# Row 2
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    device_trust_score = st.number_input(
        "Device Trust Score",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["device_trust_score"],
        step=0.01,
        format="%.2f",
        key="device_trust_score"
    )


with col2:

    risk_score_internal = st.number_input(
        "Internal Risk Score",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["risk_score_internal"],
        step=0.01,
        format="%.2f",
        key="risk_score_internal"
    )


with col3:

    corridor_risk = st.number_input(
        "Corridor Risk",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["corridor_risk"],
        step=0.01,
        format="%.2f",
        key="corridor_risk"
    )


st.divider()


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.subheader("💰 Transaction Details")


timestamp_date = st.date_input(
    "Transaction Date",
    value=datetime.now().date(),
    key="transaction_date"
)


transaction_time = st.time_input(
    "Transaction Time",
    value=datetime.now().time().replace(second=0, microsecond=0),
    key="transaction_time"
)


col1, col2, col3 = st.columns(3)


with col1:

    amount_src = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )


with col2:

    fee = st.number_input(
        "Transaction Fee",
        min_value=0.0,
        value=0.0,
        step=0.01
    )


with col3:

    channel_options = unique_values("channel")

    channel = st.selectbox(
        "Transaction Channel",
        channel_options
    )


# ============================================================
# CURRENCY INFORMATION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    source_currency_options = unique_values(
        "source_currency"
    )

    source_currency = st.selectbox(
        "Source Currency",
        source_currency_options
    )


with col2:

    dest_currency_options = unique_values(
        "dest_currency"
    )

    dest_currency = st.selectbox(
        "Destination Currency",
        dest_currency_options
    )


st.divider()


# ============================================================
# DEVICE / LOCATION / RISK INFORMATION
# ============================================================

st.subheader("🌍 Device, Location & Risk Information")


col1, col2, col3 = st.columns(3)


with col1:

    new_device_options = unique_values("new_device")

    new_device = st.selectbox(
        "New Device",
        new_device_options
    )


with col2:

    ip_country_options = unique_values("ip_country")

    ip_country = st.selectbox(
        "IP Country",
        ip_country_options
    )


with col3:

    location_mismatch_options = unique_values(
        "location_mismatch"
    )

    location_mismatch = st.selectbox(
        "Location Mismatch",
        location_mismatch_options
    )


ip_risk_score = st.slider(
    "IP Risk Score",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state["ip_risk_score"],
    step=0.01,
    key="ip_risk_score"
)


st.divider()


# ============================================================
# PREDICTION
# ============================================================

st.subheader("🔍 Fraud Detection")


predict_button = st.button(
    "Run Fraud Detection",
    type="primary",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Combine date + time correctly
    # --------------------------------------------------------

    transaction_datetime = datetime.combine(
        timestamp_date,
        transaction_time
    )


    # --------------------------------------------------------
    # Construct API payload
    # --------------------------------------------------------

    payload = {

        "customer_id": str(selected_customer),

        "timestamp": transaction_datetime.isoformat(),

        "home_country": home_country,

        "source_currency": source_currency,

        "dest_currency": dest_currency,

        "channel": channel,

        "amount_src": float(amount_src),

        "fee": float(fee),

        "new_device": new_device,

        "ip_country": ip_country,

        "location_mismatch": location_mismatch,

        "ip_risk_score": float(ip_risk_score),

        "kyc_tier": kyc_tier,

        "account_age_days": int(account_age_days),

        "device_trust_score": float(device_trust_score),

        "risk_score_internal": float(risk_score_internal),

        "corridor_risk": float(corridor_risk)
    }


    # --------------------------------------------------------
    # Send request to FastAPI
    # --------------------------------------------------------

    try:

        with st.spinner("Analysing transaction..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )


        # ----------------------------------------------------
        # Handle successful response
        # ----------------------------------------------------

        if response.status_code == 200:

            result = response.json()

            st.success(
                "Transaction successfully analysed."
            )


            # ------------------------------------------------
            # Prediction result
            # ------------------------------------------------

            st.markdown("### Prediction Result")


            result_col1, result_col2 = st.columns(2)


            with result_col1:

                if result["is_fraud"] == 1:

                    st.error(
                        "🚨 FRAUDULENT TRANSACTION"
                    )

                else:

                    st.success(
                        "✅ LEGITIMATE TRANSACTION"
                    )


            with result_col2:

                probability = (
                    result["fraud_probability"] * 100
                )

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2f}%"
                )


            # ------------------------------------------------
            # Transaction analysis
            # ------------------------------------------------

            st.markdown("### Transaction Analysis")


            metric1, metric2, metric3, metric4 = st.columns(4)


            with metric1:

                st.metric(
                    "1-Hour Velocity",
                    result.get(
                        "tnx_velocity_1h",
                        "N/A"
                    )
                )


            with metric2:

                st.metric(
                    "24-Hour Velocity",
                    result.get(
                        "tnx_velocity_24h",
                        "N/A"
                    )
                )


            with metric3:

                st.metric(
                    "Velocity Spike",
                    result.get(
                        "velocity_spike",
                        "N/A"
                    )
                )


            with metric4:

                amount_usd = result.get(
                    "amount_usd"
                )

                if amount_usd is not None:

                    st.metric(
                        "Amount (USD)",
                        f"${amount_usd:,.2f}"
                    )

                else:

                    st.metric(
                        "Amount (USD)",
                        "N/A"
                    )


            # ------------------------------------------------
            # Submitted transaction
            # ------------------------------------------------

            with st.expander(
                "View transaction submitted to API"
            ):

                st.json(payload)


        else:

            st.error(
                f"API Error ({response.status_code})"
            )

            try:

                st.json(response.json())

            except Exception:

                st.write(response.text)


    except requests.exceptions.ConnectionError:

        st.error(
            """
            Could not connect to the FastAPI server.

            Make sure your FastAPI application is running with:

            `uvicorn main.app:app --reload`
            """
        )


    except requests.exceptions.Timeout:

        st.error(
            "The API request timed out. Please try again."
        )


    except Exception as e:

        st.error(
            f"An unexpected error occurred: {e}"
        )