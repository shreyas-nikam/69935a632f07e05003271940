
# Streamlit Application Specification: AI Model Governance for Financial Professionals

## 1. Application Overview

This Streamlit application, designed for **CFA Charterholders and Investment Professionals** like **Mr. Alex Chen, a Risk Officer at FinSecure Bank**, provides a comprehensive system for AI Model Governance. It addresses the critical need to monitor AI model decisions, detect anomalous behavior, and ensure regulatory compliance in financial institutions.

The application guides Alex through a real-world workflow, allowing him to manage FinSecure Bank's AI models:
1.  **Setting up the Audit Log**: Alex establishes a foundational database for all AI decisions and alerts, a cornerstone for compliance.
2.  **Simulating Production Decisions**: Alex generates a stream of AI model decisions, including intentionally injected anomalies, to rigorously test the monitoring system's ability to catch unusual behavior.
3.  **Proactive Anomaly Detection**: Alex applies automated statistical checks to identify deviations in decision patterns, confidence, concentration, and volume, acting as "smoke detectors" for AI model risk.
4.  **Reviewing Alerts and Decisions**: Alex uses an interactive dashboard to consolidate monitoring insights, drill down into detected anomalies, review flagged decisions, and update their review status.
5.  **Generating Audit Reports**: Alex produces periodic compliance reports summarizing model activity, detected anomalies, review statuses, and regulatory compliance statements, crucial for internal stakeholders and external regulators.

The core purpose is to transform invisible AI decisions into observable, reviewable, and alertable data, enabling proactive risk management and compliance with regulations like SR 11-7, the EU AI Act Article 12, and FINRA expectations.

## 2. Code Requirements

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io # For report download

# Import all functions and classes from the source.py file
from source import AIDecisionLogger, audit_logged, generate_trading_signal, score_credit_application, \
    simulate_production_day, detect_decision_anomalies, generate_audit_report, logger # 'logger' is an instance

# --- Session State Initialization ---
def initialize_session_state():
    if 'logger' not in st.session_state:
        # Use the pre-initialized logger instance from source.py
        st.session_state['logger'] = logger
        # Ensure tables are created on first run or after reset
        st.session_state.logger._create_tables()
        st.session_state['app_initialized'] = False # Flag to run initial simulation only once

    if 'page' not in st.session_state:
        st.session_state['page'] = "Home"

    # Simulation parameters
    if 'total_simulation_days' not in st.session_state:
        st.session_state['total_simulation_days'] = 6
    if 'anomaly_trigger_day' not in st.session_state:
        st.session_state['anomaly_trigger_day'] = 4 # 1-indexed for UI, will adjust for 0-indexed in simulate_production_day
    if 'n_trading_decisions_per_day' not in st.session_state:
        st.session_state['n_trading_decisions_per_day'] = 50
    if 'n_credit_decisions_per_day' not in st.session_state:
        st.session_state['n_credit_decisions_per_day'] = 100

    # Model monitoring
    if 'models_to_monitor' not in st.session_state:
        st.session_state['models_to_monitor'] = ['MomentumSignal', 'CreditXGBoost']

    # Dashboard parameters
    if 'selected_dashboard_model' not in st.session_state:
        st.session_state['selected_dashboard_model'] = st.session_state['models_to_monitor'][0] if st.session_state['models_to_monitor'] else None
    if 'dashboard_review_period' not in st.session_state:
        st.session_state['dashboard_review_period'] = 7

    # Anomaly Detection parameters
    if 'ad_window_days' not in st.session_state:
        st.session_state['ad_window_days'] = 1
    if 'ad_baseline_days' not in st.session_state:
        st.session_state['ad_baseline_days'] = 5

    # Audit Report parameters
    if 'audit_report_period' not in st.session_state:
        st.session_state['audit_report_period'] = 7
    if 'generated_audit_report' not in st.session_state:
        st.session_state['generated_audit_report'] = {}

    # DataFrames for display (cached/updated on demand)
    if 'current_decisions_df' not in st.session_state:
        st.session_state['current_decisions_df'] = pd.DataFrame()
    if 'current_alerts_df' not in st.session_state:
        st.session_state['current_alerts_df'] = pd.DataFrame()


initialize_session_state()

# --- Main Application Layout ---
st.sidebar.title("AI Model Governance Platform")
st.sidebar.header("Navigation")
st.session_state['page'] = st.sidebar.selectbox(
    "Choose a section",
    ["Home", "Simulation & Data Generation", "Anomaly Detection", "Risk Officer Dashboard", "Audit Report"]
)

# --- Common functions for data retrieval and state update ---
def update_decision_review_status(decision_id, new_status, review_notes):
    """Updates the review status and notes for a specific decision.
    Invokes: AIDecisionLogger.conn.execute, AIDecisionLogger.conn.commit
    Updates: st.session_state['current_decisions_df']
    """
    try:
        current_time = datetime.now().isoformat()
        st.session_state.logger.conn.execute(
            '''UPDATE decisions SET review_status = ?, review_notes = ?, review_date = ?, reviewed_by = ? WHERE id = ?''',
            (new_status, review_notes, current_time, "Alex Chen", decision_id)
        )
        st.session_state.logger.conn.commit()
        st.success(f"Decision {decision_id} updated to '{new_status}'.")
        # Invalidate current decisions data to trigger refresh on next access
        st.session_state['current_decisions_df'] = pd.DataFrame()
    except Exception as e:
        st.error(f"Error updating decision {decision_id}: {e}")

def acknowledge_alert(alert_id, acknowledged_by="Alex Chen"):
    """Acknowledges a specific alert.
    Invokes: AIDecisionLogger.conn.execute, AIDecisionLogger.conn.commit
    Updates: st.session_state['current_alerts_df']
    """
    try:
        st.session_state.logger.conn.execute(
            '''UPDATE alerts SET acknowledged = 1, acknowledged_by = ? WHERE id = ?''',
            (acknowledged_by, alert_id)
        )
        st.session_state.logger.conn.commit()
        st.success(f"Alert {alert_id} acknowledged by {acknowledged_by}.")
        # Invalidate current alerts data to trigger refresh on next access
        st.session_state['current_alerts_df'] = pd.DataFrame()
    except Exception as e:
        st.error(f"Error acknowledging alert {alert_id}: {e}")

# --- Page: Home ---
if st.session_state['page'] == "Home":
    st.title("AI Model Governance: Proactive Monitoring and Anomaly Detection for Financial Professionals")

    st.markdown(f"")
    st.markdown(f"## Introduction: Mr. Alex Chen, Risk Officer at FinSecure Bank")
    st.markdown(f"Welcome to FinSecure Bank, a leading financial institution leveraging AI for critical operations like automated trading signal generation and credit approvals. You are **Mr. Alex Chen, a seasoned Risk Officer** responsible for ensuring that the bank's AI models operate within acceptable risk parameters, comply with internal governance policies (e.g., Model Risk Management guidelines like SR 11-7), and meet external regulatory requirements (e.g., EU AI Act Article 12).")
    st.markdown(f"FinSecure's AI models generate thousands of decisions daily. Reviewing each decision manually is impossible and inefficient. Your primary challenge is to implement a robust, scalable system that continuously monitors these AI decisions, automatically detects anomalous behavior, and provides the necessary tools for investigation and reporting, thereby proactively mitigating potential financial losses or regulatory penalties.")
    st.markdown(f"This application walks you through the practical steps Alex would take to build and utilize such a system, focusing on:")
    st.markdown(f"1.  **Establishing a comprehensive audit log** for all AI decisions.")
    st.markdown(f"2.  **Implementing non-invasive logging** to integrate with existing models.")
    st.markdown(f"3.  **Simulating real-world AI decisions**, including carefully injected anomalies.")
    st.markdown(f"4.  **Developing automated anomaly detection algorithms** to flag deviations from normal behavior.")
    st.markdown(f"5.  **Creating an interactive monitoring dashboard** for quick risk assessment.")
    st.markdown(f"6.  **Generating a periodic audit report** for compliance and stakeholder communication.")

    st.markdown(f"---")
    st.markdown(f"### 2. Building the Foundation: The AI Decision Audit Log Database")
    st.markdown(f"As a Risk Officer, Alex knows that the cornerstone of effective AI governance is a comprehensive, immutable audit log. Regulatory frameworks like SR 11-7 and the EU AI Act Article 12 explicitly mandate the automatic recording of events and decisions by high-risk AI systems. This log serves as the single source of truth for every AI decision, capturing crucial metadata, inputs, outputs, and explanations. Without such a log, FinSecure Bank cannot reconstruct past decisions, investigate incidents, or demonstrate compliance.")
    st.markdown(f"Alex needs a structured database to store:")
    st.markdown(f"1.  **Decision records**: Every AI model's output, along with its context.")
    st.markdown(f"2.  **Alert records**: Any detected anomalies or deviations from expected behavior.")
    st.markdown(f"He will use SQLite for simplicity in this demonstration, but in a real-world scenario, this could be a more robust enterprise database.")
    st.markdown(f"**Concept:** Structured Logging Schema")
    st.markdown(f"The schema for the `decisions` table must capture all relevant details to reconstruct and review any AI decision. Key fields include: `timestamp`, `model_name`, `model_version`, `decision_type`, `input_features`, `input_hash`, `prediction`, `confidence`, `explanation`, `ticker`, `sector`, `portfolio_id`, `user_id`, `review_status`, `anomaly_flag`.")
    st.markdown(f"Similarly, the `alerts` table tracks detected anomalies with `alert_type`, `severity`, `description`, and `model_name`.")

    st.markdown(f"---")
    st.markdown(f"### 3. Non-Invasive Decision Capture with Decorators")
    st.markdown(f"Integrating audit logging into existing production AI models can be complex. Model development teams often prioritize prediction accuracy and performance. The governance team, led by Alex, needs to ensure logging without directly modifying the sensitive, validated model code. Modifying core model logic introduces risk and requires re-validation.")
    st.markdown(f"To address this, Alex decided to implement a **non-invasive logging wrapper using a Python decorator**. This architectural pattern allows the governance team to 'decorate' any prediction function, injecting logging capabilities without altering the original function's source code. This separation of concerns is a best practice in robust MLOps.")
    st.markdown(f"**Concept:** Decorator Pattern for Non-Invasive Logging")
    st.markdown(f"A decorator `@audit_logged` will:")
    st.markdown(f"1.  Take a model's prediction function as input.")
    st.markdown(f"2.  Wrap it with a new function that calls the original prediction function.")
    st.markdown(f"3.  Extract relevant metadata from the prediction's inputs and outputs.")
    st.markdown(f"4.  Log this metadata to the `AIDecisionLogger`.")
    st.markdown(f"5.  Return the original prediction result, ensuring the model's behavior is unchanged.")

    if not st.session_state['app_initialized']:
        with st.spinner("Initializing audit log database and simulating initial data..."):
            # Initial simulation upon first load
            for day_idx in range(st.session_state['total_simulation_days']):
                is_anomaly = (day_idx == st.session_state['anomaly_trigger_day'] - 1)
                simulate_production_day(st.session_state.logger,
                                        n_trading=st.session_state['n_trading_decisions_per_day'],
                                        n_credit=st.session_state['n_credit_decisions_per_day'],
                                        anomaly_day=is_anomaly)
            st.session_state['app_initialized'] = True
        st.success("Initial data simulation complete! Navigate to 'Risk Officer Dashboard' to see monitoring insights, or 'Anomaly Detection' to run checks!")
    else:
        st.info("Application is initialized with simulated data. Navigate to other sections!")

# --- Page: Simulation & Data Generation ---
elif st.session_state['page'] == "Simulation & Data Generation":
    st.title("Simulating AI Decisions with Injected Anomalies")

    st.markdown(f"As Mr. Alex Chen, to validate the anomaly detection capabilities, you need to test the system with real-world scenarios, including days where models behave abnormally. Simulating production decisions allows you to control the environment and inject specific types of anomalies to ensure the monitoring system effectively flags them. This proactive testing is crucial for gaining confidence in the governance framework before live incidents occur.")
    st.markdown(f"You will simulate several days of trading signals and credit approvals, intentionally introducing anomalous behavior on one of these days to see if the detection system catches it.")
    st.markdown(f"**Concept:** Synthetic Data Generation with Injected Anomalies")
    st.markdown(f"The simulation will generate:")
    st.markdown(f"-   **Normal decisions**: Following expected patterns for `momentum_12m`, `fico`, `dti`.")
    st.markdown(f"-   **Injected anomalies**:")
    st.markdown(f"    -   **Trading Model Anomaly**: A sudden shift in recommendations, e.g., an unusually high proportion of 'SELL' signals. This is simulated by forcing `momentum_12m` to a low value for many decisions on an 'anomaly day'.")
    st.markdown(f"    -   **Credit Model Anomaly**: A subtle shift in approval criteria, e.g., approving a higher number of applicants with low FICO scores. This is simulated by lowering the `fico` score for some approved applications on the 'anomaly day'.")

    st.subheader("Simulation Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state['total_simulation_days'] = st.number_input(
            "Total simulation days",
            min_value=1, value=st.session_state['total_simulation_days'], step=1
        )
    with col2:
        st.session_state['anomaly_trigger_day'] = st.number_input(
            "Anomaly day (1-indexed)",
            min_value=1, max_value=st.session_state['total_simulation_days'],
            value=st.session_state['anomaly_trigger_day'], step=1
        )
    col3, col4 = st.columns(2)
    with col3:
        st.session_state['n_trading_decisions_per_day'] = st.number_input(
            "Trading decisions per day",
            min_value=10, value=st.session_state['n_trading_decisions_per_day'], step=10
        )
    with col4:
        st.session_state['n_credit_decisions_per_day'] = st.number_input(
            "Credit decisions per day",
            min_value=10, value=st.session_state['n_credit_decisions_per_day'], step=10
        )


    st.markdown(f"---")
    st.subheader("Generate Decisions")
    if st.button("Run New Simulation and Generate Decisions"):
        with st.spinner(f"Clearing existing data and simulating {st.session_state['total_simulation_days']} days of AI decisions..."):
            # Clear existing data before new simulation to avoid duplicates
            st.session_state.logger.conn.execute("DELETE FROM decisions")
            st.session_state.logger.conn.execute("DELETE FROM alerts")
            st.session_state.logger.conn.commit()

            for day_idx in range(st.session_state['total_simulation_days']):
                is_anomaly = (day_idx == st.session_state['anomaly_trigger_day'] - 1)
                simulate_production_day(st.session_state.logger,
                                        n_trading=st.session_state['n_trading_decisions_per_day'],
                                        n_credit=st.session_state['n_credit_decisions_per_day'],
                                        anomaly_day=is_anomaly)
            st.session_state['app_initialized'] = True # Confirm re-initialization
        st.success("Decision simulation complete and data logged!")

    st.subheader("Simulated Decisions Overview")
    st.session_state['current_decisions_df'] = st.session_state.logger.get_decisions(days=st.session_state['total_simulation_days'], limit=5000)
    if not st.session_state['current_decisions_df'].empty:
        st.markdown(f"Total logged decisions over the last {st.session_state['total_simulation_days']} days: **{len(st.session_state['current_decisions_df'])}**")
        st.dataframe(st.session_state['current_decisions_df'].sort_values('timestamp', ascending=False).head(10))
    else:
        st.info("No decisions logged yet. Run the simulation to generate data.")

# --- Page: Anomaly Detection ---
elif st.session_state['page'] == "Anomaly Detection":
    st.title("Proactive Anomaly Detection in Decision Streams")

    st.markdown(f"Alex's primary role is to proactively identify and mitigate risks. Manual review of every decision is impossible. He needs an automated system to constantly monitor the decision streams and alert him when model behavior deviates from established norms. This is where anomaly detection algorithms come into play, providing the 'smoke detectors' for AI model risk.")
    st.markdown(f"He will implement four essential anomaly checks. For each check, a **threshold** is defined. If the deviation exceeds this threshold, an alert is logged with a severity level. This aligns with SR 11-7's requirement for ongoing monitoring and proactive risk mitigation.")
    st.markdown(f"**Concept:** Anomaly Detection Algorithms with Statistical Thresholds")

    st.subheader("Anomaly Detection Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state['ad_window_days'] = st.number_input(
            "Recent Window Days",
            min_value=1, value=st.session_state['ad_window_days'], step=1,
            help="Number of recent days to compare against baseline."
        )
    with col2:
        st.session_state['ad_baseline_days'] = st.number_input(
            "Baseline Window Days",
            min_value=1, value=st.session_state['ad_baseline_days'], step=1,
            help="Number of historical days for baseline (excluding recent window)."
        )

    st.subheader("Anomaly Checks & Formulas")

    st.markdown(f"**1. Decision Distribution Shift:**")
    st.markdown(f"This check compares the distribution of `prediction` categories in the recent `window_days` against a `baseline_days` period.")
    st.markdown(r"$$ \Delta P_d = |P_{\text{{recent}}}(d) - P_{\text{{baseline}}}(d)| $$")
    st.markdown(r"where $P_{\text{{recent}}}(d)$ is the proportion of decision category $d$ in the recent window, and $P_{\text{{baseline}}}(d)$ is the proportion in the baseline window.")
    st.markdown(f"An alert is triggered if $\Delta P_d > 0.20$ (20 percentage points) for any decision category $d$.")

    st.markdown(f"**2. Concentration Risk:**")
    st.markdown(f"This identifies if the model is disproportionately focusing on a single entity (`ticker` for trading signals, not applicable for credit applications in this demo).")
    st.markdown(f"For the recent window, calculate the maximum proportion of decisions attributed to any single entity:")
    st.markdown(r"$$ \text{{Max Concentration}} = \max_{{\text{{entity}}}} \left( \frac{{\text{{Count of decisions for entity}}}}{{\text{{Total decisions in recent window}}}} \right) $$")
    st.markdown(r"where 'entity' refers to a unique `ticker` for trading signals (e.g., AAPL, MSFT).")
    st.markdown(f"An alert is triggered if $\text{{Max Concentration}} > 0.30$ (30%).")

    st.markdown(f"**3. Confidence Anomaly:**")
    st.markdown(f"This monitors shifts in the model's average prediction confidence.")
    st.markdown(r"$$ \Delta \text{{Confidence}} = |\text{{mean Confidence}}_{\text{{recent}}} - \text{{mean Confidence}}_{\text{{baseline}}}| $$")
    st.markdown(r"where $\text{{mean Confidence}}_{\text{{recent}}}$ is the average confidence in the recent window, and $\text{{mean Confidence}}_{\text{{baseline}}}$ is the average confidence in the baseline.")
    st.markdown(f"An alert is triggered if $\Delta \text{{Confidence}} > 0.10$.")

    st.markdown(f"**4. Volume Anomaly:**")
    st.markdown(f"This check detects abnormal daily decision counts.")
    st.markdown(f"Let $\text{{Daily Count}}_{\text{{recent}}}$ be the average daily decisions in the recent window, and $\text{{Daily Count}}_{\text{{baseline}}}$ be the average daily decisions in the baseline.")
    st.markdown(f"An alert is triggered if $\text{{Daily Count}}_{\text{{recent}}} > 2 \times \text{{Daily Count}}_{\text{{baseline}}}$ (more than double) or $\text{{Daily Count}}_{\text{{recent}}} < 0.3 \times \text{{Daily Count}}_{\text{{baseline}}}$ (less than 30% of baseline).")

    st.markdown(f"---")
    st.subheader("Run Anomaly Detection")
    run_ad_button = st.button("Run Anomaly Detection for All Models")
    if run_ad_button:
        st.session_state['current_alerts_df'] = pd.DataFrame() # Clear previous alerts view
        with st.spinner("Running anomaly detection checks..."):
            all_detected_alerts = []
            for model in st.session_state['models_to_monitor']:
                # Invokes: detect_decision_anomalies
                alerts = detect_decision_anomalies(
                    st.session_state.logger,
                    model,
                    window_days=st.session_state['ad_window_days'],
                    baseline_days=st.session_state['ad_baseline_days']
                )
                all_detected_alerts.extend(alerts)
        if all_detected_alerts:
            st.success(f"Anomaly detection complete! {len(all_detected_alerts)} new alerts detected and logged.")
        else:
            st.info("No new anomalies detected.")

    st.subheader("Recent Alerts")
    # Invokes: AIDecisionLogger.get_alerts
    st.session_state['current_alerts_df'] = st.session_state.logger.get_alerts(
        days=st.session_state['total_simulation_days'] + st.session_state['ad_baseline_days'] # Ensure all simulated days are covered
    )

    if not st.session_state['current_alerts_df'].empty:
        # Display alerts and allow acknowledgment
        st.dataframe(st.session_state['current_alerts_df'].sort_values('timestamp', ascending=False))

        st.markdown(f"**Review and Acknowledge Alerts**")
        col_alert_id, col_ack_button = st.columns([1, 1])
        alert_to_ack = col_alert_id.number_input("Enter Alert ID to Acknowledge",
                                                  min_value=st.session_state['current_alerts_df']['id'].min(),
                                                  max_value=st.session_state['current_alerts_df']['id'].max(),
                                                  value=st.session_state['current_alerts_df']['id'].min(),
                                                  step=1, key='alert_id_input')
        if col_ack_button.button("Acknowledge Selected Alert"):
            if alert_to_ack in st.session_state['current_alerts_df']['id'].values:
                # Invokes: acknowledge_alert
                acknowledge_alert(alert_to_ack)
                st.experimental_rerun() # Rerun to refresh the displayed alerts
            else:
                st.warning(f"Alert ID {alert_to_ack} not found in the current list.")
    else:
        st.info("No alerts found in the database for the specified period. Run a simulation and anomaly detection first.")

# --- Page: Risk Officer Dashboard ---
elif st.session_state['page'] == "Risk Officer Dashboard":
    st.title("The Risk Officer's Monitoring Dashboard")

    st.markdown(f"With a continuous stream of decisions and potential alerts, Alex needs a consolidated 'Risk Officer Review Dashboard.' This dashboard must provide a high-level overview of AI model activity, highlight detected anomalies, show decision distributions, and identify any concentration risks. This unified view enables Alex to quickly grasp the current state of FinSecure's AI models, prioritize his review tasks, and respond promptly to critical issues, fulfilling the 'ongoing monitoring' aspect of SR 11-7.")
    st.markdown(f"**Concept:** Data Aggregation and Visualization for Insights")

    st.sidebar.subheader("Dashboard Controls")
    st.session_state['selected_dashboard_model'] = st.sidebar.selectbox(
        "Select Model",
        st.session_state['models_to_monitor'],
        index=st.session_state['models_to_monitor'].index(st.session_state['selected_dashboard_model'])
    )
    st.session_state['dashboard_review_period'] = st.sidebar.slider(
        "Review Period (days)",
        min_value=1, max_value=st.session_state['total_simulation_days'],
        value=min(st.session_state['dashboard_review_period'], st.session_state['total_simulation_days']), step=1
    )

    model_name = st.session_state['selected_dashboard_model']
    days = st.session_state['dashboard_review_period']

    # Invokes: AIDecisionLogger.get_decisions, AIDecisionLogger.get_alerts
    decisions_df = st.session_state.logger.get_decisions(model_name, days=days, limit=10000)
    alerts_df = st.session_state.logger.get_alerts(model_name, days=days)

    st.subheader(f"Dashboard for: {model_name} (Last {days} days)")

    if decisions_df.empty:
        st.warning(f"No decisions recorded for {model_name} in the last {days} days. Please run the simulation first.")
    else:
        # --- Summary Statistics ---
        st.markdown(f"---")
        st.subheader("Summary Statistics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        total_decisions = len(decisions_df)
        decisions_per_day = total_decisions / max(1, days)
        pending_reviews = len(decisions_df[decisions_df['review_status'] == 'pending'])
        flagged_decisions = len(decisions_df[decisions_df['anomaly_flag'] == 1]) # From initial simulation marking
        total_alerts = len(alerts_df)
        high_severity_alerts = len(alerts_df[alerts_df['severity'] == 'HIGH'])
        unacknowledged_alerts = len(alerts_df[alerts_df['acknowledged'] == 0])


        col1.metric("Total Decisions", total_decisions)
        col2.metric("Decisions/Day (Avg)", f"{decisions_per_day:.0f}")
        col3.metric("Pending Reviews", pending_reviews)
        col4.metric("Flagged Decisions", flagged_decisions)
        col5.metric("Total Alerts", total_alerts)
        col6.metric("Unacknowledged Alerts", unacknowledged_alerts)


        # --- Visualizations ---
        st.markdown(f"---")
        st.subheader("Decision Distribution Over Time")
        decisions_df['date'] = decisions_df['timestamp'].dt.normalize()
        daily_dist = decisions_df.groupby(['date', 'prediction']).size().unstack(fill_value=0)
        daily_dist_pct = daily_dist.div(daily_dist.sum(axis=1), axis=0) # Convert to proportions

        fig_dist_time = px.bar(
            daily_dist_pct,
            x=daily_dist_pct.index,
            y=daily_dist_pct.columns,
            title='Daily Decision Distribution (%)',
            labels={'x': 'Date', 'value': 'Proportion', 'variable': 'Decision'}, # 'variable' is default for unstacked columns
            color_discrete_sequence=px.colors.qualitative.Vivid,
            height=450
        )
        fig_dist_time.update_layout(yaxis_tickformat=".0%")
        # Highlight anomaly days if distribution shift alerts are present
        for _, alert_row in alerts_df[alerts_df['alert_type'] == 'distribution_shift'].iterrows():
            alert_date = pd.to_datetime(alert_row['timestamp']).normalize()
            if alert_date in daily_dist_pct.index:
                fig_dist_time.add_vline(x=alert_date.timestamp() * 1000, line_width=2, line_dash="dash", line_color="red", annotation_text="Shift Alert", annotation_position="top right")
        st.plotly_chart(fig_dist_time, use_container_width=True)


        st.markdown(f"---")
        st.subheader("Confidence Distribution")
        if 'confidence' in decisions_df.columns:
            clean_confidence = pd.to_numeric(decisions_df['confidence'], errors='coerce').dropna()
            if not clean_confidence.empty:
                fig_conf = px.histogram(
                    clean_confidence,
                    nbins=20,
                    title='Distribution of Prediction Confidence Scores',
                    labels={'value': 'Confidence Score', 'count': 'Frequency'},
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    height=450
                )
                # Optionally highlight confidence anomalies
                for _, alert_row in alerts_df[alerts_df['alert_type'] == 'confidence_shift'].iterrows():
                    fig_conf.add_annotation(
                        x=clean_confidence.mean(), y=fig_conf.layout.yaxis.range[1]*0.9,
                        text=f"Confidence Shift Alert", showarrow=True, arrowhead=1,
                        font=dict(color="red"), bgcolor="rgba(255,0,0,0.1)"
                    )
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.info("No valid confidence data available for this model in the selected period.")
        else:
            st.info("No 'confidence' column available for this model.")

        st.markdown(f"---")
        st.subheader("Concentration Risk")
        # Concentration risk only applies meaningfully to trading signals (tickers) for this demo
        if 'ticker' in decisions_df.columns and model_name == 'MomentumSignal':
            ticker_counts = decisions_df['ticker'].value_counts(normalize=True).head(10) # Top 10 tickers
            if not ticker_counts.empty:
                fig_concentration = px.bar(
                    ticker_counts,
                    x=ticker_counts.values,
                    y=ticker_counts.index,
                    orientation='h',
                    title=f'Top Ticker Concentration for {model_name} (%)',
                    labels={'x': 'Proportion of Decisions', 'y': 'Ticker'},
                    color_discrete_sequence=px.colors.qualitative.D3,
                    height=450
                )
                fig_concentration.update_layout(xaxis_tickformat=".0%")
                # Highlight if max concentration alert exists
                if not alerts_df[alerts_df['alert_type'] == 'concentration_risk'].empty:
                    fig_concentration.add_vline(x=0.30, line_width=2, line_dash="dot", line_color="red", annotation_text="Concentration Threshold (30%)", annotation_position="bottom right")
                st.plotly_chart(fig_concentration, use_container_width=True)
            else:
                st.info("No ticker data available for this model in the selected period.")
        else:
            st.info("Concentration risk analysis not applicable or no ticker data for this model.")

        st.markdown(f"---")
        st.subheader("Alert Timeline")
        if not alerts_df.empty:
            severity_map = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
            # Create a severity_order column for consistent Y-axis ordering
            alerts_df['severity_order'] = alerts_df['severity'].map({'LOW': 1, 'MEDIUM': 2, 'HIGH': 3})

            fig_alerts = px.scatter(
                alerts_df.sort_values('severity_order'), # Sort to ensure higher severity is on top if points overlap
                x='timestamp',
                y='severity', # Plot actual severity text on Y-axis
                color='severity',
                title='Alert Timeline by Severity',
                labels={'timestamp': 'Time', 'severity': 'Severity Level'},
                color_discrete_map=severity_map,
                height=450
            )
            # Ensure Y-axis displays severity levels in a specific order
            fig_alerts.update_layout(yaxis=dict(categoryorder='array', categoryarray=['LOW', 'MEDIUM', 'HIGH']))
            fig_alerts.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig_alerts, use_container_width=True)
        else:
            st.info("No alerts in this period for this model.")

        st.markdown(f"---")
        st.subheader("Flagged Decisions for Review")
        st.markdown(f"Alex needs to review decisions that have been flagged due to anomalous behavior or are otherwise pending review.")

        review_status_filter = st.selectbox(
            "Filter Decisions by Review Status",
            ['All', 'pending', 'reviewed', 'acknowledged'],
            key='dashboard_review_filter'
        )

        filtered_decisions_df = decisions_df.copy()
        if review_status_filter != 'All':
            filtered_decisions_df = filtered_decisions_df[filtered_decisions_df['review_status'] == review_status_filter]

        if not filtered_decisions_df.empty:
            st.dataframe(filtered_decisions_df.sort_values('timestamp', ascending=False))

            st.markdown(f"**Update Review Status for a Decision**")
            col_dec_id, col_new_status = st.columns(2)
            decision_id_to_update = col_dec_id.number_input("Decision ID",
                                                              min_value=filtered_decisions_df['id'].min(),
                                                              max_value=filtered_decisions_df['id'].max(),
                                                              value=filtered_decisions_df['id'].iloc[0] if not filtered_decisions_df.empty else 1,
                                                              step=1, key='dashboard_decision_id_input')
            new_review_status = col_new_status.selectbox("New Status", ['pending', 'reviewed', 'acknowledged'], key='dashboard_new_status_select')
            review_notes = st.text_area("Review Notes", key='dashboard_review_notes_input')

            if st.button("Update Decision Review Status", key='dashboard_update_decision_button'):
                if decision_id_to_update in decisions_df['id'].values:
                    # Invokes: update_decision_review_status
                    update_decision_review_status(decision_id_to_update, new_review_status, review_notes)
                    st.experimental_rerun() # Rerun to refresh the displayed decisions
                else:
                    st.warning(f"Decision ID {decision_id_to_update} not found in current view.")
        else:
            st.info("No decisions to display for review based on current filters and model activity.")

# --- Page: Audit Report ---
elif st.session_state['page'] == "Audit Report":
    st.title("Generating a Periodic AI Model Audit Report")

    st.markdown(f"Beyond daily monitoring, Alex, as a Risk Officer, is responsible for providing formal, periodic compliance reports to FinSecure Bank's risk committee and external regulators. These reports summarize AI model activity, highlight detected anomalies, detail review statuses, and affirm compliance with relevant regulations like SR 11-7 and the EU AI Act. This formal documentation is essential for demonstrating robust AI governance and accountability.")
    st.markdown(f"**Concept:** Structured Reporting for Compliance")

    st.sidebar.subheader("Report Controls")
    st.session_state['audit_report_period'] = st.sidebar.slider(
        "Report Period (days)",
        min_value=1, max_value=st.session_state['total_simulation_days'],
        value=min(st.session_state['audit_report_period'], st.session_state['total_simulation_days']), step=1
    )

    if st.button("Generate Audit Report"):
        with st.spinner(f"Generating audit report for the last {st.session_state['audit_report_period']} days..."):
            # Invokes: generate_audit_report
            st.session_state['generated_audit_report'] = generate_audit_report(
                st.session_state.logger,
                period_days=st.session_state['audit_report_period']
            )
        st.success("Audit report generated!")

    if st.session_state['generated_audit_report']:
        report_data = st.session_state['generated_audit_report']
        st.header(report_data['report_title'])
        st.markdown(f"**Period**: {report_data['period']}")
        st.markdown(f"**Generated**: {report_data['generation_date']}")

        st.markdown(f"### Executive Summary")
        for key, value in report_data['executive_summary'].items():
            st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")

        st.markdown(f"### Model Summaries")
        if not report_data['model_summaries']:
            st.info("No model summaries for the period.")
        for model, stats in report_data['model_summaries'].items():
            st.markdown(f"#### Model: {model}")
            for stat_key, stat_value in stats.items():
                st.markdown(f"  - **{stat_key.replace('_', ' ').title()}**: {stat_value}")

        st.markdown(f"### Regulatory Compliance Status")
        for reg, status in report_data['regulatory_compliance'].items():
            st.markdown(f"- **{reg.replace('_', ' ').title()}**: {status}")

        st.markdown(f"### Sign-Off")
        for role, details in report_data['sign_off'].items():
            st.markdown(f"- **{role}**: {details['name']} (Date: {details['date']})")
            st.markdown(f"  Signature: __________________________")

        # Option to download the report as Markdown/Text
        report_text = ""
        report_text += f"# {report_data['report_title']}\n"
        report_text += f"**Period**: {report_data['period']}\n"
        report_text += f"**Generated**: {report_data['generation_date']}\n\n"
        report_text += "## Executive Summary\n"
        for key, value in report_data['executive_summary'].items():
            report_text += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        report_text += "\n## Model Summaries\n"
        if not report_data['model_summaries']:
            report_text += "No model summaries for the period.\n"
        for model, stats in report_data['model_summaries'].items():
            report_text += f"\n### Model: {model}\n"
            for stat_key, stat_value in stats.items():
                report_text += f"  - **{stat_key.replace('_', ' ').title()}**: {stat_value}\n"
        report_text += "\n## Regulatory Compliance Status\n"
        for reg, status in report_data['regulatory_compliance'].items():
            report_text += f"- **{reg.replace('_', ' ').title()}**: {status}\n"
        report_text += "\n## Sign-Off\n"
        for role, details in report_data['sign_off'].items():
            report_text += f"- **{role}**: {details['name']} (Date: {details['date']})\n"
            report_text += "  Signature: __________________________\n\n"

        st.download_button(
            label="Download Report as Text",
            data=report_text,
            file_name=f"AI_Model_Audit_Report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
    else:
        st.info("Click 'Generate Audit Report' to view its content for the selected period.")
```
