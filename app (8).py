from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "car_price_model.joblib"
REFERENCE_YEAR = 2026  # must match the notebook: car_age = 2026 - year

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def inr(x: float) -> str:
    """Format a number with Indian digit grouping, e.g. ₹4,50,000."""
    s = str(int(round(x)))
    if len(s) <= 3:
        return f"₹{s}"
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "₹" + ",".join(groups + [tail])


def build_features(bundle, row: dict) -> pd.DataFrame:
    """Rebuild the exact training features, then scale."""
    cat_cols = bundle["cat_cols"]
    encoder = bundle["encoder"]

    num = pd.DataFrame(
        [
            {
                "km_driven": row["km_driven"],
                "mileage": row["mileage"],
                "engine": row["engine"],
                "max_power": row["max_power"],
                "seats": row["seats"],
                "car_age": REFERENCE_YEAR - row["year"],
                "power_per_engine": row["max_power"] / row["engine"],
            }
        ]
    )

    cats = pd.DataFrame([{c: row[c] for c in cat_cols}])
    encoded = pd.DataFrame(
        encoder.transform(cats[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=num.index,
    )

    X = pd.concat([num, encoded], axis=1)
    X = X.reindex(columns=bundle["feature_columns"], fill_value=0)
    return pd.DataFrame(bundle["scaler"].transform(X), columns=X.columns)


def predict(bundle, row: dict):
    X = build_features(bundle, row)
    model = bundle["model"]
    uses_log = bundle["uses_log_target"]

    def back(v):
        return np.expm1(v) if uses_log else v

    X = X.values  # the notebook fit the model on a scaled numpy array
    price = float(max(back(model.predict(X)[0]), 0))

    low = high = None
    if hasattr(model, "estimators_"):  # Random Forest: spread across trees
        tree_preds = np.array([back(t.predict(X)[0]) for t in model.estimators_])
        low, high = np.percentile(tree_preds, [10, 90])
    return price, low, high


# ---------- UI ----------
st.title("🚗 Used Car Price Predictor")
st.caption("Estimate a used car's resale price from its specs. Trained on 8K+ CarDekho listings.")

if not MODEL_PATH.exists():
    st.error("`car_price_model.joblib` not found. Put it in the same folder as `app.py`.")
    st.stop()

bundle = load_bundle()
cat_cols = bundle["cat_cols"]
options = {c: list(cats) for c, cats in zip(cat_cols, bundle["encoder"].categories_)}

with st.sidebar:
    st.header("About")
    st.write(f"**Model:** {bundle['model_name'].replace('_', ' ').title()}")
    st.write("**Test R²:** 0.92")
    st.write("**Avg error:** ~₹74K")
    st.caption("Prices are in Indian rupees (₹). Model metrics come from the training notebook.")


def default_index(values, preferred):
    return values.index(preferred) if preferred in values else 0


with st.form("car_form"):
    c1, c2 = st.columns(2)

    with c1:
        brand = st.selectbox("Brand", options["brand"], index=default_index(options["brand"], "Maruti"))
        year = st.number_input("Manufacturing year", min_value=1990, max_value=REFERENCE_YEAR, value=2015, step=1)
        km_driven = st.number_input("Kilometers driven", min_value=0, max_value=1_000_000, value=70_000, step=5_000)
        fuel = st.selectbox("Fuel", options["fuel"], index=default_index(options["fuel"], "Diesel"))
        transmission = st.selectbox(
            "Transmission", options["transmission"], index=default_index(options["transmission"], "Manual")
        )

    with c2:
        seller_type = st.selectbox(
            "Seller type", options["seller_type"], index=default_index(options["seller_type"], "Individual")
        )
        owner = st.selectbox("Owner", options["owner"], index=default_index(options["owner"], "First Owner"))
        mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=45.0, value=19.4, step=0.1)
        engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1248, step=50)
        max_power = st.number_input("Max power (bhp)", min_value=30.0, max_value=500.0, value=82.0, step=1.0)

    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10, 14], index=2)
    submitted = st.form_submit_button("Predict price", use_container_width=True, type="primary")

if submitted:
    row = {
        "brand": brand,
        "year": int(year),
        "km_driven": float(km_driven),
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": float(mileage),
        "engine": float(engine),
        "max_power": float(max_power),
        "seats": float(seats),
    }
    price, low, high = predict(bundle, row)

    st.divider()
    st.metric("Estimated price", inr(price))
    if low is not None:
        st.caption(f"Model spread: {inr(low)} to {inr(high)} (10th–90th percentile across the forest's trees).")
    st.caption("This is an estimate from historical listings, not a valuation.")
