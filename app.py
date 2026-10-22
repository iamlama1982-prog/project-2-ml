"""Streamlit decision-support demo for the Telco churn project."""

import pandas as pd
import streamlit as st

from src.demo import (
    action_for_score,
    add_scores,
    churn_scores,
    customer_drivers,
    load_model_and_demo_split,
    priority_band,
    scenario_customers,
)


def select_index(options: list, value) -> int:
    """Return a safe selectbox index for a preset value."""

    if value in options:
        return options.index(value)
    return 0


def format_percentage(value: float) -> str:
    """Format a decimal risk score as a percentage for table display."""

    return f"{value:.1%}"


def format_money(value: float) -> str:
    """Format a charge value consistently in the queue table."""

    return f"{value:.2f}"


def format_senior(value: int) -> str:
    """Turn the dataset's 0/1 senior flag into readable UI text."""

    if value == 1:
        return "Yes"
    return "No"


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    """Load or recreate the final model once per Streamlit session."""

    return load_model_and_demo_split()


st.set_page_config(
    page_title="Retention Control Room",
    page_icon="☎️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
        .hero {
            padding: 1.35rem 1.5rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #0f172a, #172554);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 {margin: 0 0 .25rem 0; font-size: 2.0rem;}
        .hero p {margin: 0; color: #dbeafe; font-size: 1.02rem;}
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: .08em;
            font-size: .72rem;
            font-weight: 700;
            color: #5eead4;
            margin-bottom: .25rem;
        }
        .small-note {font-size: .86rem; color: #64748b;}
        .action-box {
            padding: 1rem 1.1rem;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            background: #f8fafc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

model, threshold, X_test, y_test, test_ids, model_source = get_model_bundle()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Customer Service Decision Support</div>
        <h1>Retention Control Room</h1>
        <p>Turn churn predictions into a manageable call queue — who needs attention first, what workload does that create, and where should an agent use judgement?</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Model guardrails")
    st.metric("Held-out churn recall", "78.3%")
    st.metric("Held-out precision", "50.4%")
    st.metric("Decision threshold", f"{threshold:.2f}")
    st.caption(model_source)
    st.divider()
    st.markdown(
        "**Use for:** prioritising low-cost retention outreach.\n\n"
        "**Do not use for:** automatic discounts, contract decisions or promises that "
        "the displayed score is a perfectly calibrated probability."
    )

queue_tab, customer_tab = st.tabs([
    "☎️ Today's retention queue",
    "👤 Score one customer",
])


with queue_tab:
    st.subheader("Build the day's retention workload")
    st.write(
        "Start with the held-out demonstration portfolio, or upload a customer file using "
        "the same columns as the IBM Telco data. The model ranks customers by churn score; "
        "your available contact capacity decides how much of the queue can actually be worked."
    )

    source = st.radio(
        "Customer source",
        ["Held-out demo portfolio", "Upload a customer CSV"],
        horizontal=True,
    )

    actual_outcomes = None

    if source == "Upload a customer CSV":
        uploaded_file = st.file_uploader(
            "Upload customer data",
            type=["csv"],
            help="The file may include customerID and Churn; those are not used as model features.",
        )

        if uploaded_file is None:
            st.info("Upload a file to build a live retention queue. Until then, switch to the demo portfolio to explore the tool.")
            st.stop()

        portfolio = pd.read_csv(uploaded_file)
        if "Churn" in portfolio.columns:
            actual_outcomes = portfolio["Churn"].copy()
    else:
        portfolio = X_test.copy()
        portfolio.insert(0, "customerID", test_ids.loc[portfolio.index].values)
        actual_outcomes = y_test.loc[portfolio.index].copy()

    try:
        scored = add_scores(model, portfolio, threshold)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    flagged = scored[scored["Risk score"] >= threshold].copy()
    flagged_count = len(flagged)

    if flagged_count > 0:
        default_capacity = min(300, flagged_count)
        contact_capacity = st.slider(
            "How many proactive contacts can the retention team make?",
            min_value=1,
            max_value=flagged_count,
            value=default_capacity,
        )
    else:
        contact_capacity = 0
        st.warning("No customers in this file are above the model's contact threshold.")

    average_minutes = st.slider(
        "Average agent minutes per proactive contact",
        min_value=2,
        max_value=20,
        value=5,
    )

    queue = flagged.head(contact_capacity).copy()
    workload_hours = (len(queue) * average_minutes) / 60
    coverage = 0.0
    if flagged_count > 0:
        coverage = len(queue) / flagged_count

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Customers scored", f"{len(scored):,}")
    metric_2.metric("Flagged for retention", f"{flagged_count:,}")
    metric_3.metric("Today's queue", f"{len(queue):,}", f"{coverage:.0%} of flagged")
    metric_4.metric("Agent workload", f"{workload_hours:.1f} h", f"at {average_minutes} min/contact")

    if actual_outcomes is not None and len(queue) > 0:
        outcome_lookup = actual_outcomes.to_dict()
        queue_actual = []

        for row_index in queue.index:
            queue_actual.append(outcome_lookup.get(row_index))

        queue["Actual outcome"] = queue_actual
        actual_churners = int((actual_outcomes == "Yes").sum())
        caught_churners = int((queue["Actual outcome"] == "Yes").sum())
        false_alarms = int((queue["Actual outcome"] == "No").sum())

        caught_share = 0.0
        if actual_churners > 0:
            caught_share = caught_churners / actual_churners

        st.markdown("#### Back-test: what would this capacity have achieved?")
        back_1, back_2, back_3 = st.columns(3)
        back_1.metric("Known churners reached", f"{caught_churners:,} / {actual_churners:,}", f"{caught_share:.1%} of churners")
        back_2.metric("Extra contacts", f"{false_alarms:,}", "customers who actually stayed")
        if len(queue) > 0:
            queue_hit_rate = caught_churners / len(queue)
        else:
            queue_hit_rate = 0.0
        back_3.metric("Queue hit rate", f"{queue_hit_rate:.1%}", "demo outcomes only")

        st.caption(
            "The back-test uses the known outcomes in the held-out demonstration data. "
            "A real future queue would not know these outcomes yet."
        )

    if flagged_count > contact_capacity:
        deferred = flagged_count - contact_capacity
        st.warning(
            f"Capacity gap: {deferred:,} additional customers are above the model threshold "
            "but would not be contacted at this capacity. The table below therefore keeps "
            "the highest model scores first."
        )
    elif flagged_count > 0:
        st.success("Current capacity can cover every customer above the model threshold.")

    st.markdown("#### Prioritised agent queue")

    display_columns = []
    for column in [
        "customerID",
        "Priority",
        "Risk score",
        "tenure",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "Actual outcome",
    ]:
        if column in queue.columns:
            display_columns.append(column)

    queue_display = queue[display_columns].copy()
    if "Risk score" in queue_display.columns:
        queue_display["Risk score"] = queue_display["Risk score"].map(format_percentage)
    if "MonthlyCharges" in queue_display.columns:
        queue_display["MonthlyCharges"] = queue_display["MonthlyCharges"].map(format_money)

    st.dataframe(
        queue_display,
        use_container_width=True,
        hide_index=True,
    )

    export_queue = queue.copy()
    export_queue["Risk score"] = export_queue["Risk score"].round(4)
    st.download_button(
        "Download prioritised queue as CSV",
        data=export_queue.to_csv(index=False).encode("utf-8"),
        file_name="retention_priority_queue.csv",
        mime="text/csv",
    )

    st.markdown(
        '<p class="small-note">Management note: the score is most defensible as a ranking signal. '
        'The project did not separately calibrate probabilities, and the model cannot see recent complaints, outages, competitor offers or other live service context.</p>',
        unsafe_allow_html=True,
    )


with customer_tab:
    st.subheader("Score a customer live")
    st.write(
        "Choose one of the presentation scenarios, edit anything you like, then score the "
        "customer. The result includes an operational next step and the strongest model "
        "signals for this specific row."
    )

    scenarios = scenario_customers()
    scenario_name = st.selectbox("Start from a scenario", list(scenarios.keys()))
    preset = scenarios[scenario_name]

    with st.form("customer_form"):
        st.markdown("##### Customer and account")
        col_1, col_2, col_3 = st.columns(3)

        gender_options = ["Female", "Male"]
        with col_1:
            gender = st.selectbox(
                "Gender",
                gender_options,
                index=select_index(gender_options, preset["gender"]),
            )
            senior = st.selectbox(
                "Senior citizen",
                [0, 1],
                index=select_index([0, 1], preset["SeniorCitizen"]),
                format_func=format_senior,
            )
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=int(preset["tenure"]),
            )

        with col_2:
            yes_no = ["Yes", "No"]
            partner = st.selectbox(
                "Partner",
                yes_no,
                index=select_index(yes_no, preset["Partner"]),
            )
            dependents = st.selectbox(
                "Dependents",
                yes_no,
                index=select_index(yes_no, preset["Dependents"]),
            )
            contract_options = ["Month-to-month", "One year", "Two year"]
            contract = st.selectbox(
                "Contract",
                contract_options,
                index=select_index(contract_options, preset["Contract"]),
            )

        with col_3:
            monthly_charges = st.number_input(
                "Monthly charges",
                min_value=0.0,
                max_value=200.0,
                value=float(preset["MonthlyCharges"]),
                step=1.0,
            )
            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=10000.0,
                value=float(preset["TotalCharges"]),
                step=10.0,
            )
            paperless = st.selectbox(
                "Paperless billing",
                yes_no,
                index=select_index(yes_no, preset["PaperlessBilling"]),
            )

        st.markdown("##### Services and payment")
        service_1, service_2, service_3 = st.columns(3)

        with service_1:
            phone_service = st.selectbox(
                "Phone service",
                yes_no,
                index=select_index(yes_no, preset["PhoneService"]),
            )
            multiple_options = ["No", "Yes", "No phone service"]
            multiple_lines = st.selectbox(
                "Multiple lines",
                multiple_options,
                index=select_index(multiple_options, preset["MultipleLines"]),
            )
            internet_options = ["DSL", "Fiber optic", "No"]
            internet = st.selectbox(
                "Internet service",
                internet_options,
                index=select_index(internet_options, preset["InternetService"]),
            )

        internet_addon_options = ["Yes", "No", "No internet service"]
        with service_2:
            online_security = st.selectbox(
                "Online security",
                internet_addon_options,
                index=select_index(internet_addon_options, preset["OnlineSecurity"]),
            )
            online_backup = st.selectbox(
                "Online backup",
                internet_addon_options,
                index=select_index(internet_addon_options, preset["OnlineBackup"]),
            )
            device_protection = st.selectbox(
                "Device protection",
                internet_addon_options,
                index=select_index(internet_addon_options, preset["DeviceProtection"]),
            )
            tech_support = st.selectbox(
                "Tech support",
                internet_addon_options,
                index=select_index(internet_addon_options, preset["TechSupport"]),
            )

        with service_3:
            streaming_tv = st.selectbox(
                "Streaming TV",
                internet_addon_options,
                index=select_index(internet_addon_options, preset["StreamingTV"]),
            )
            streaming_movies = st.selectbox(
                "Streaming movies",
                internet_addon_options,
                index=select_index(internet_addon_options, preset["StreamingMovies"]),
            )
            payment_options = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            payment_method = st.selectbox(
                "Payment method",
                payment_options,
                index=select_index(payment_options, preset["PaymentMethod"]),
            )

        submitted = st.form_submit_button("Score customer", use_container_width=True)

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])

        score = float(churn_scores(model, customer)[0])
        priority = priority_band(score, threshold)
        action = action_for_score(score, threshold)
        positive_drivers, negative_drivers = customer_drivers(model, customer)

        result_1, result_2, result_3 = st.columns([1, 1, 2])
        result_1.metric("Model churn score", f"{score:.1%}")
        result_2.metric("Operational priority", priority)
        if score >= threshold:
            result_3.warning("Above the model's retention-contact threshold")
        else:
            result_3.success("Below the model's retention-contact threshold")

        st.markdown(
            f'<div class="action-box"><strong>Suggested next step</strong><br>{action}</div>',
            unsafe_allow_html=True,
        )

        driver_left, driver_right = st.columns(2)
        with driver_left:
            st.markdown("**Strongest pushes towards churn**")
            if positive_drivers:
                for driver in positive_drivers:
                    st.write("• " + driver)
            else:
                st.write("No strong positive contribution for this row.")

        with driver_right:
            st.markdown("**Strongest pushes towards staying**")
            if negative_drivers:
                for driver in negative_drivers:
                    st.write("• " + driver)
            else:
                st.write("No strong negative contribution for this row.")

        st.caption(
            "The driver list is calculated from this customer's transformed Logistic Regression "
            "feature contributions. It explains the model's score; it does not prove that any "
            "feature caused the customer to churn."
        )