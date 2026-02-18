This README provides a comprehensive overview of the **QuLab: Lab 39: Audit Logging Demo** Streamlit application, designed to showcase a robust AI Model Governance platform.

---

# QuLab: Lab 39: Audit Logging Demo - AI Model Governance Platform

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title

**QuLab: Lab 39: Audit Logging Demo - AI Model Governance Platform**

## Description

This Streamlit application simulates a real-world scenario where **Mr. Alex Chen, a Risk Officer at FinSecure Bank**, is tasked with implementing a comprehensive AI model governance framework. The application demonstrates critical aspects of monitoring, anomaly detection, and reporting for AI models used in high-stakes financial operations like automated trading signal generation and credit approvals.

The core challenge addressed is how to effectively monitor thousands of daily AI decisions, proactively detect anomalous behavior, and provide tools for investigation and regulatory compliance without manually reviewing every decision.

**Key objectives demonstrated by this lab project:**

1.  **Establishing a comprehensive audit log**: Creating a robust, immutable database to record every AI decision with crucial metadata.
2.  **Implementing non-invasive logging**: Integrating logging capabilities into existing AI models using Python decorators, avoiding direct modification of core model logic.
3.  **Simulating real-world AI decisions**: Generating synthetic data, including carefully injected anomalies (e.g., shifts in trading recommendations or credit approval criteria).
4.  **Developing automated anomaly detection algorithms**: Implementing statistical checks to flag deviations from normal model behavior.
5.  **Creating an interactive monitoring dashboard**: Providing a consolidated view for risk officers to quickly assess AI model activity, review flagged decisions, and manage alerts.
6.  **Generating a periodic audit report**: Producing formal, compliance-ready reports for stakeholders and regulators, fulfilling requirements like SR 11-7 and the EU AI Act.

This application serves as a practical guide for financial professionals to understand and implement proactive AI model risk management and governance.

## Features

The application is structured into several interactive sections, each demonstrating a key aspect of AI model governance:

*   **Home**: An introduction to the problem, the persona of Mr. Alex Chen, and the foundational concepts of audit logging and non-invasive decision capture using decorators.
*   **Simulation & Data Generation**:
    *   Configurable parameters for simulating daily trading and credit decisions.
    *   Ability to inject specific anomalies (e.g., unusual trading signals, lenient credit approvals) on a designated "anomaly day".
    *   Generates and logs thousands of synthetic AI decisions into the audit database.
*   **Anomaly Detection**:
    *   Implements four key statistical anomaly checks:
        1.  **Decision Distribution Shift**: Detects changes in the proportion of different decision outcomes.
        2.  **Concentration Risk**: Identifies if a model is disproportionately focusing on a single entity (e.g., a specific stock ticker).
        3.  **Confidence Anomaly**: Monitors shifts in the model's average prediction confidence.
        4.  **Volume Anomaly**: Flags abnormal daily decision counts.
    *   Allows configuration of "recent window" and "baseline window" for comparison.
    *   Logs detected anomalies as alerts with severity levels.
    *   Provides an interface to review and acknowledge alerts.
*   **Risk Officer Dashboard**:
    *   An interactive dashboard for selected AI models (Trading Signal, Credit Approval).
    *   Displays summary statistics (total decisions, decisions/day, pending reviews, alerts).
    *   Visualizations: Daily decision distribution over time, confidence score distribution, concentration risk by entity (for trading models), and an alert timeline.
    *   Includes a section to review and update the status of flagged decisions.
*   **Audit Report**:
    *   Generates a comprehensive, periodic audit report summarizing model activity, anomalies, and compliance status.
    *   Report includes executive summary, model-specific statistics, regulatory compliance affirmations, and sign-off sections.
    *   Provides an option to download the generated report in markdown format.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/quolab-audit-logging-demo.git
    cd quolab-audit-logging-demo
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The application relies on several Python libraries. Create a `requirements.txt` file in your project root with the following content:

    ```
    streamlit>=1.0.0
    pandas>=1.0.0
    numpy>=1.0.0
    plotly>=5.0.0
    ```

    Then, install them:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure `source.py` exists**:
    This project expects a file named `source.py` in the same directory as `app.py`. This file contains the core logic for the `AIDecisionLogger` class, the `@audit_logged` decorator, `simulate_production_day`, `detect_decision_anomalies`, and `generate_audit_report` functions. You will need to have this file populated with the necessary implementation details for the application to run correctly.

    A placeholder for `source.py` would look like:

    ```python
    # source.py
    import streamlit as st
    import pandas as pd
    import numpy as np
    import sqlite3
    import hashlib
    import json
    from datetime import datetime, timedelta
    from functools import wraps

    class AIDecisionLogger:
        def __init__(self, db_name='ai_audit_log.db'):
            self.db_name = db_name
            self.conn = sqlite3.connect(db_name, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self._create_tables()

        def _create_tables(self):
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model_name TEXT,
                    model_version TEXT,
                    decision_type TEXT,
                    input_features TEXT, -- JSON string
                    input_hash TEXT,    -- Hash of input_features
                    prediction TEXT,
                    confidence REAL,
                    explanation TEXT,   -- JSON string or text
                    ticker TEXT,        -- Applicable for trading signals
                    sector TEXT,        -- Applicable for trading signals
                    portfolio_id TEXT,  -- Applicable for trading signals
                    user_id TEXT,       -- Applicable for credit decisions
                    review_status TEXT DEFAULT 'pending',
                    anomaly_flag INTEGER DEFAULT 0, -- 1 if initially flagged by simulation logic
                    review_notes TEXT,
                    review_date TEXT,
                    reviewed_by TEXT
                )
            ''')
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model_name TEXT,
                    alert_type TEXT,
                    severity TEXT, -- e.g., LOW, MEDIUM, HIGH
                    description TEXT,
                    decision_ids TEXT, -- JSON list of decision IDs related to this alert
                    acknowledged INTEGER DEFAULT 0, -- 0 for unacknowledged, 1 for acknowledged
                    acknowledged_by TEXT,
                    acknowledged_date TEXT
                )
            ''')
            self.conn.commit()

        def log_decision(self, model_name, model_version, decision_type, input_features,
                         prediction, confidence, explanation=None, ticker=None,
                         sector=None, portfolio_id=None, user_id=None, anomaly_flag=0):
            timestamp = datetime.now().isoformat()
            input_json = json.dumps(input_features)
            input_hash = hashlib.sha256(input_json.encode('utf-8')).hexdigest()
            explanation_json = json.dumps(explanation) if explanation else None

            self.cursor.execute('''
                INSERT INTO decisions (timestamp, model_name, model_version, decision_type,
                                    input_features, input_hash, prediction, confidence,
                                    explanation, ticker, sector, portfolio_id, user_id,
                                    anomaly_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, model_name, model_version, decision_type,
                  input_json, input_hash, prediction, confidence,
                  explanation_json, ticker, sector, portfolio_id, user_id,
                  anomaly_flag))
            self.conn.commit()
            return self.cursor.lastrowid

        def log_alert(self, model_name, alert_type, severity, description, related_decision_ids=None):
            timestamp = datetime.now().isoformat()
            decision_ids_json = json.dumps(related_decision_ids) if related_decision_ids else "[]"
            self.cursor.execute('''
                INSERT INTO alerts (timestamp, model_name, alert_type, severity, description, decision_ids)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, model_name, alert_type, severity, description, decision_ids_json))
            self.conn.commit()
            return self.cursor.lastrowid

        def get_decisions(self, model_name=None, days=None, limit=None):
            query = "SELECT * FROM decisions"
            conditions = []
            params = []

            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)

            if days:
                start_date = (datetime.now() - timedelta(days=days)).isoformat()
                conditions.append("timestamp >= ?")
                params.append(start_date)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql_query(query, self.conn, params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Convert 'input_features' and 'explanation' back from JSON string if needed
                # df['input_features'] = df['input_features'].apply(json.loads)
                # df['explanation'] = df['explanation'].apply(lambda x: json.loads(x) if x else None)
            return df

        def get_alerts(self, model_name=None, days=None):
            query = "SELECT * FROM alerts"
            conditions = []
            params = []

            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)

            if days:
                start_date = (datetime.now() - timedelta(days=days)).isoformat()
                conditions.append("timestamp >= ?")
                params.append(start_date)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"

            df = pd.read_sql_query(query, self.conn, params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # df['decision_ids'] = df['decision_ids'].apply(json.loads)
            return df

    # Initialize the logger instance once
    logger = AIDecisionLogger()

    # --- Decorator for Non-Invasive Logging ---
    def audit_logged(model_name, model_version, decision_type):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract input features - assuming the first arg is features dict for simplicity
                input_features = args[0] if args else {}

                # Call the original prediction function
                result = func(*args, **kwargs)

                # Assuming result is a dict with 'prediction', 'confidence', 'explanation'
                # Adjust based on your actual model output structure
                prediction = result.get('prediction')
                confidence = result.get('confidence', 0.0)
                explanation = result.get('explanation')

                # Extract specific attributes based on decision_type
                ticker = input_features.get('ticker') if decision_type == 'trading_signal' else None
                sector = input_features.get('sector') if decision_type == 'trading_signal' else None
                portfolio_id = input_features.get('portfolio_id') if decision_type == 'trading_signal' else None
                user_id = input_features.get('user_id') if decision_type == 'credit_approval' else None

                # Log the decision using the global logger instance
                logger.log_decision(
                    model_name=model_name,
                    model_version=model_version,
                    decision_type=decision_type,
                    input_features=input_features,
                    prediction=prediction,
                    confidence=confidence,
                    explanation=explanation,
                    ticker=ticker,
                    sector=sector,
                    portfolio_id=portfolio_id,
                    user_id=user_id
                )
                return result
            return wrapper
        return decorator

    # --- Simulated Models (decorated) ---
    @audit_logged(model_name='MomentumSignal', model_version='1.0', decision_type='trading_signal')
    def momentum_trading_model(features):
        momentum_12m = features['momentum_12m']
        if momentum_12m > 0.05:
            prediction = 'BUY'
            confidence = min(0.95, 0.5 + momentum_12m)
        elif momentum_12m < -0.05:
            prediction = 'SELL'
            confidence = min(0.95, 0.5 - momentum_12m)
        else:
            prediction = 'HOLD'
            confidence = 0.5
        explanation = f"Momentum 12m: {momentum_12m:.2f}. Decision based on threshold."
        return {'prediction': prediction, 'confidence': confidence, 'explanation': explanation}

    @audit_logged(model_name='CreditXGBoost', model_version='2.1', decision_type='credit_approval')
    def credit_scoring_model(features):
        fico = features['fico']
        dti = features['dti']
        loan_amount = features['loan_amount']

        # Simplified logic for demo
        score = (fico * 0.4) - (dti * 0.3) - (loan_amount / 10000 * 0.3)
        if score > 200:
            prediction = 'APPROVED'
            confidence = min(0.9, (score - 200) / 100 + 0.6)
        else:
            prediction = 'REJECTED'
            confidence = min(0.9, (200 - score) / 100 + 0.6)

        explanation = f"FICO: {fico}, DTI: {dti:.2f}, Loan: {loan_amount}. Score: {score:.2f}."
        return {'prediction': prediction, 'confidence': confidence, 'explanation': explanation}

    # --- Simulation Function ---
    def simulate_production_day(logger_instance, n_trading, n_credit, anomaly_day=False):
        current_date = (datetime.now() - timedelta(days=st.session_state['total_simulation_days'] - 1) + timedelta(days=st.session_state['current_sim_day_offset'] if 'current_sim_day_offset' in st.session_state else 0)).strftime('%Y-%m-%d')
        st.session_state['current_sim_day_offset'] = st.session_state.get('current_sim_day_offset', 0) + 1

        # Simulate Trading Decisions
        for _ in range(n_trading):
            features = {
                'ticker': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'GS']),
                'sector': np.random.choice(['Tech', 'Finance', 'Energy', 'Healthcare']),
                'portfolio_id': f"PF{np.random.randint(100, 999)}",
                'momentum_12m': np.random.normal(0.02, 0.08) # Normal distribution around 2%
            }
            if anomaly_day:
                # Force more SELL signals on anomaly day for trading model
                if np.random.rand() < 0.4: # 40% chance to force low momentum
                    features['momentum_12m'] = np.random.uniform(-0.15, -0.06)
            momentum_trading_model(features)

        # Simulate Credit Decisions
        for _ in range(n_credit):
            features = {
                'fico': np.random.randint(600, 850),
                'dti': np.random.uniform(0.1, 0.5),
                'loan_amount': np.random.randint(5000, 50000),
                'user_id': f"U{np.random.randint(10000, 99999)}"
            }
            if anomaly_day:
                # On anomaly day, approve more low FICO scores
                if np.random.rand() < 0.3: # 30% chance to force lower FICO for "approved" case
                    features['fico'] = np.random.randint(500, 650) # FICO in a range that would normally be rejected or marginal

            credit_scoring_model(features)
        
        st.session_state.logger.conn.execute(f"UPDATE decisions SET timestamp = REPLACE(timestamp, SUBSTR(timestamp, 1, 10), '{current_date}') WHERE timestamp LIKE '{datetime.now().strftime('%Y-%m-%d')}%'")
        st.session_state.logger.conn.execute(f"UPDATE alerts SET timestamp = REPLACE(timestamp, SUBSTR(timestamp, 1, 10), '{current_date}') WHERE timestamp LIKE '{datetime.now().strftime('%Y-%m-%d')}%'")
        st.session_state.logger.conn.commit()


    # --- Anomaly Detection Function ---
    def detect_decision_anomalies(logger_instance, model_name, window_days, baseline_days):
        end_date = datetime.now()
        window_start = end_date - timedelta(days=window_days)
        baseline_start = window_start - timedelta(days=baseline_days)

        recent_df = logger_instance.get_decisions(model_name=model_name, days=window_days)
        baseline_df = logger_instance.get_decisions(model_name=model_name, days=baseline_days, limit=100000) # Get more baseline data

        alerts = []

        if recent_df.empty:
            return alerts # Cannot detect anomalies without recent data

        # Ensure datetime conversion for filtering
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
        baseline_df['timestamp'] = pd.to_datetime(baseline_df['timestamp'])

        # Filter for exact windows
        recent_df = recent_df[(recent_df['timestamp'] >= window_start) & (recent_df['timestamp'] < end_date)]
        baseline_df = baseline_df[(baseline_df['timestamp'] >= baseline_start) & (baseline_df['timestamp'] < window_start)]

        if recent_df.empty or baseline_df.empty:
            return alerts # Need both recent and baseline for comparison

        # 1. Decision Distribution Shift
        recent_dist = recent_df['prediction'].value_counts(normalize=True)
        baseline_dist = baseline_df['prediction'].value_counts(normalize=True)

        for p_type in set(recent_dist.index).union(baseline_dist.index):
            delta = abs(recent_dist.get(p_type, 0) - baseline_dist.get(p_type, 0))
            if delta > 0.20: # 20 percentage points
                alerts.append(logger_instance.log_alert(
                    model_name=model_name,
                    alert_type='distribution_shift',
                    severity='HIGH',
                    description=f"Significant shift in '{p_type}' decision distribution. Recent: {recent_dist.get(p_type,0):.2f}, Baseline: {baseline_dist.get(p_type,0):.2f}. Delta: {delta:.2f}.",
                    related_decision_ids=recent_df['id'].tolist()
                ))

        # 2. Concentration Risk (only for trading_signal)
        if model_name == 'MomentumSignal' and 'ticker' in recent_df.columns:
            ticker_concentration = recent_df['ticker'].value_counts(normalize=True).max()
            if ticker_concentration > 0.30: # 30% concentration
                alerts.append(logger_instance.log_alert(
                    model_name=model_name,
                    alert_type='concentration_risk',
                    severity='MEDIUM',
                    description=f"High concentration risk detected. Max ticker concentration: {ticker_concentration:.2f}.",
                    related_decision_ids=recent_df['id'].tolist()
                ))

        # 3. Confidence Anomaly
        if 'confidence' in recent_df.columns and 'confidence' in baseline_df.columns:
            recent_confidence_mean = pd.to_numeric(recent_df['confidence'], errors='coerce').mean()
            baseline_confidence_mean = pd.to_numeric(baseline_df['confidence'], errors='coerce').mean()
            if not pd.isna(recent_confidence_mean) and not pd.isna(baseline_confidence_mean):
                delta_confidence = abs(recent_confidence_mean - baseline_confidence_mean)
                if delta_confidence > 0.10: # 0.10 absolute difference
                    alerts.append(logger_instance.log_alert(
                        model_name=model_name,
                        alert_type='confidence_shift',
                        severity='MEDIUM',
                        description=f"Shift in average confidence score. Recent: {recent_confidence_mean:.2f}, Baseline: {baseline_confidence_mean:.2f}. Delta: {delta_confidence:.2f}.",
                        related_decision_ids=recent_df['id'].tolist()
                    ))

        # 4. Volume Anomaly
        recent_daily_counts = recent_df.groupby(recent_df['timestamp'].dt.date).size()
        baseline_daily_counts = baseline_df.groupby(baseline_df['timestamp'].dt.date).size()

        if not recent_daily_counts.empty and not baseline_daily_counts.empty:
            avg_recent_volume = recent_daily_counts.mean()
            avg_baseline_volume = baseline_daily_counts.mean()

            if avg_recent_volume > 2 * avg_baseline_volume:
                alerts.append(logger_instance.log_alert(
                    model_name=model_name,
                    alert_type='volume_anomaly',
                    severity='HIGH',
                    description=f"Abnormally high decision volume. Recent avg: {avg_recent_volume:.0f}, Baseline avg: {avg_baseline_volume:.0f}. (>2x baseline)",
                    related_decision_ids=recent_df['id'].tolist()
                ))
            elif avg_recent_volume < 0.3 * avg_baseline_volume:
                alerts.append(logger_instance.log_alert(
                    model_name=model_name,
                    alert_type='volume_anomaly',
                    severity='HIGH',
                    description=f"Abnormally low decision volume. Recent avg: {avg_recent_volume:.0f}, Baseline avg: {avg_baseline_volume:.0f}. (<0.3x baseline)",
                    related_decision_ids=recent_df['id'].tolist()
                ))
        return alerts

    # --- Audit Report Generation Function ---
    def generate_audit_report(logger_instance, period_days=7):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        period_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        all_decisions = logger_instance.get_decisions(days=period_days, limit=100000)
        all_alerts = logger_instance.get_alerts(days=period_days)

        report = {
            "report_title": "AI Model Governance Audit Report",
            "period": period_str,
            "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "executive_summary": {},
            "model_summaries": {},
            "regulatory_compliance": {
                "SR 11-7 (Model Risk Management)": "Compliant: Comprehensive logging, ongoing monitoring, independent review facilitated.",
                "EU AI Act Article 12 (Log-keeping)": "Compliant: Automatic recording of events and decisions by high-risk AI systems implemented."
            },
            "sign_off": {
                "Risk Officer": {"name": "Alex Chen", "date": datetime.now().strftime('%Y-%m-%d')},
                "Head of AI/ML": {"name": "[Signature Required]", "date": ""}
            }
        }

        # Executive Summary
        total_decisions = len(all_decisions)
        total_alerts = len(all_alerts)
        unacknowledged_alerts = len(all_alerts[all_alerts['acknowledged'] == 0])
        flagged_decisions = len(all_decisions[all_decisions['anomaly_flag'] == 1])
        decisions_pending_review = len(all_decisions[all_decisions['review_status'] == 'pending'])

        report['executive_summary'] = {
            "Total AI Decisions Logged": total_decisions,
            "Total Alerts Detected": total_alerts,
            "Unacknowledged Alerts": unacknowledged_alerts,
            "Decisions Flagged by Simulation Anomaly": flagged_decisions,
            "Decisions Pending Manual Review": decisions_pending_review,
            "Key Observations": "AI models operated generally within expected parameters, with identified anomalies proactively logged and monitored. All high-severity alerts have been acknowledged and are under investigation." if unacknowledged_alerts == 0 else "Some high-severity alerts remain unacknowledged and require immediate attention."
        }

        # Model Summaries
        models = all_decisions['model_name'].unique() if not all_decisions.empty else []
        for model in models:
            model_decisions = all_decisions[all_decisions['model_name'] == model]
            model_alerts = all_alerts[all_alerts['model_name'] == model]

            total_model_decisions = len(model_decisions)
            total_model_alerts = len(model_alerts)
            high_severity_model_alerts = len(model_alerts[model_alerts['severity'] == 'HIGH'])
            model_unacknowledged_alerts = len(model_alerts[model_alerts['acknowledged'] == 0])

            # Decision Distribution
            if not model_decisions.empty:
                decision_dist = model_decisions['prediction'].value_counts(normalize=True).to_dict()
                decision_dist_str = ", ".join([f"{k}: {v:.1%}" for k,v in decision_dist.items()])
            else:
                decision_dist_str = "No decisions"

            report['model_summaries'][model] = {
                "Total Decisions": total_model_decisions,
                "Total Alerts": total_model_alerts,
                "High Severity Alerts": high_severity_model_alerts,
                "Unacknowledged Alerts": model_unacknowledged_alerts,
                "Decision Distribution": decision_dist_str,
                "Average Confidence": f"{model_decisions['confidence'].mean():.2f}" if 'confidence' in model_decisions.columns and not model_decisions['confidence'].isna().all() else "N/A"
            }
        return report
    ```

## Usage

1.  **Start the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser (usually at `http://localhost:8501`).

2.  **Navigate the application:**
    Use the **sidebar** on the left to switch between different sections:
    *   **Home**: Read the project introduction and concepts. The application performs an initial data simulation when first loaded.
    *   **Simulation & Data Generation**:
        *   Adjust parameters like total simulation days, anomaly trigger day, and daily decision counts.
        *   Click "Run New Simulation and Generate Decisions" to populate the audit log database. This will clear previous data.
    *   **Anomaly Detection**:
        *   Configure the "Recent Window Days" and "Baseline Window Days" for comparison.
        *   Click "Run Anomaly Detection for All Models" to execute the predefined checks.
        *   Review detected alerts and use the input fields to acknowledge them.
    *   **Risk Officer Dashboard**:
        *   Select a model and review period using the sidebar controls.
        *   Explore summary statistics, interactive charts (decision distribution, confidence, concentration risk, alert timeline).
        *   Filter and review individual decisions, updating their status and adding notes.
    *   **Audit Report**:
        *   Set the "Report Period" in the sidebar.
        *   Click "Generate Audit Report" to produce a summary of AI model activity and compliance.
        *   Download the report in markdown format.

## Project Structure

```
├── app.py                  # Main Streamlit application script
├── source.py               # Contains AIDecisionLogger, audit_logged decorator,
|                           # simulation logic, anomaly detection, and report generation functions
├── requirements.txt        # List of Python dependencies
└── README.md               # Project README file
```

## Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building interactive web applications and dashboards.
*   **Pandas**: For data manipulation, analysis, and data frame operations.
*   **NumPy**: For numerical operations, especially in data generation and statistical calculations.
*   **Plotly / Plotly Express**: For creating rich, interactive data visualizations within the dashboard.
*   **SQLite3**: A lightweight, file-based database used for persistent storage of audit logs (decisions and alerts).

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and ensure the code adheres to existing style guides.
4.  Write clear and concise commit messages.
5.  Push your changes to your fork.
6.  Open a Pull Request to the `main` branch of this repository.

## License

This project is licensed under the MIT License - see the LICENSE file (if applicable) for details.

## Contact

For questions, feedback, or collaborations, please contact:

*   **QuantUniversity**: [info@quantuniversity.com](mailto:info@quantuniversity.com)
*   **Project Repository**: [https://github.com/your-username/quolab-audit-logging-demo](https://github.com/your-username/quolab-audit-logging-demo) (Replace with actual repo link)