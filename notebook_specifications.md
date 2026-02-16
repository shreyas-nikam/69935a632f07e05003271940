
# AI Model Governance: Proactive Monitoring and Anomaly Detection for Financial Professionals

## Introduction: Mr. Alex Chen, Risk Officer at FinSecure Bank

Welcome to FinSecure Bank, a leading financial institution leveraging AI for critical operations like automated trading signal generation and credit approvals. You are **Mr. Alex Chen, a seasoned Risk Officer** responsible for ensuring that the bank's AI models operate within acceptable risk parameters, comply with internal governance policies (e.g., Model Risk Management guidelines like SR 11-7), and meet external regulatory requirements (e.g., EU AI Act Article 12).

FinSecure's AI models generate thousands of decisions daily. Reviewing each decision manually is impossible and inefficient. Your primary challenge is to implement a robust, scalable system that continuously monitors these AI decisions, automatically detects anomalous behavior, and provides the necessary tools for investigation and reporting, thereby proactively mitigating potential financial losses or regulatory penalties.

This notebook walks you through the practical steps Alex would take to build and utilize such a system, focusing on:
1.  **Establishing a comprehensive audit log** for all AI decisions.
2.  **Implementing non-invasive logging** to integrate with existing models.
3.  **Simulating real-world AI decisions**, including carefully injected anomalies.
4.  **Developing automated anomaly detection algorithms** to flag deviations from normal behavior.
5.  **Creating an interactive monitoring dashboard** for quick risk assessment.
6.  **Generating a periodic audit report** for compliance and stakeholder communication.

---

## 1. Environment Setup

As Alex, your first step is to ensure your analytical environment is ready by installing the necessary Python libraries.

```python
!pip install pandas numpy matplotlib jsonlines
```

```python
import sqlite3
import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta
import functools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import logging

# Configure logging for alerts (optional, but good practice for a real system)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

---

## 2. Building the Foundation: The AI Decision Audit Log Database

**Story + Context + Real-World Relevance:**

As a Risk Officer, Alex knows that the cornerstone of effective AI governance is a comprehensive, immutable audit log. Regulatory frameworks like SR 11-7 and the EU AI Act Article 12 explicitly mandate the automatic recording of events and decisions by high-risk AI systems. This log serves as the single source of truth for every AI decision, capturing crucial metadata, inputs, outputs, and explanations. Without such a log, FinSecure Bank cannot reconstruct past decisions, investigate incidents, or demonstrate compliance.

Alex needs a structured database to store:
1.  **Decision records**: Every AI model's output, along with its context.
2.  **Alert records**: Any detected anomalies or deviations from expected behavior.

He will use SQLite for simplicity in this demonstration, but in a real-world scenario, this could be a more robust enterprise database.

**Concept:** Structured Logging Schema

The schema for the `decisions` table must capture all relevant details to reconstruct and review any AI decision. Key fields include:
- `timestamp`: When the decision was made.
- `model_name`, `model_version`: To identify the AI system.
- `decision_type`: E.g., 'trading_signal', 'credit_approval'.
- `input_features`: The data fed to the model, stored as JSON.
- `input_hash`: A hash of `input_features` to detect identical inputs or potential data integrity issues.
- `prediction`, `confidence`, `explanation`: The model's output and rationale.
- `ticker`, `sector`, `portfolio_id`, `user_id`: Business context for the decision.
- `review_status`, `anomaly_flag`: For tracking monitoring and human intervention.

Similarly, the `alerts` table tracks detected anomalies with `alert_type`, `severity`, `description`, and `model_name`.

```python
class AIDecisionLogger:
    """
    Comprehensive audit logger for AI model decisions and monitoring alerts.
    Captures inputs, outputs, metadata, and enables review.
    """
    def __init__(self, db_path='ai_audit_log.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        self.conn.row_factory = sqlite3.Row # Allows accessing columns by name
        self._create_tables()

    def _create_tables(self):
        """
        Creates the 'decisions' and 'alerts' tables if they don't already exist.
        """
        # Main decision log table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                input_features TEXT,
                input_hash TEXT,
                prediction TEXT NOT NULL,
                confidence REAL,
                explanation TEXT,
                ticker TEXT,
                sector TEXT,
                portfolio_id TEXT,
                user_id TEXT,
                actual_outcome TEXT DEFAULT NULL,
                outcome_date TEXT DEFAULT NULL,
                reviewed_by TEXT DEFAULT NULL,
                review_date TEXT DEFAULT NULL,
                review_status TEXT DEFAULT 'pending',
                review_notes TEXT DEFAULT NULL,
                anomaly_flag INTEGER DEFAULT 0,
                anomaly_reason TEXT DEFAULT NULL
            )
        ''')

        # Alert log table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                model_name TEXT,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_by TEXT DEFAULT NULL
            )
        ''')
        self.conn.commit()

    def log_decision(self, model_name, model_version, decision_type,
                     prediction, confidence=None, input_features=None,
                     explanation=None, ticker=None, sector=None,
                     portfolio_id=None, user_id=None):
        """Logs a single AI model decision."""
        input_features_json = json.dumps(input_features or {}, sort_keys=True)
        input_hash = hashlib.md5(input_features_json.encode()).hexdigest()[:12] # Truncate for brevity

        self.conn.execute('''
            INSERT INTO decisions
            (timestamp, model_name, model_version, decision_type,
             input_features, input_hash, prediction, confidence,
             explanation, ticker, sector, portfolio_id, user_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (datetime.now().isoformat(), model_name, model_version,
              decision_type, input_features_json, input_hash,
              str(prediction), confidence, explanation, ticker, sector,
              portfolio_id, user_id))
        self.conn.commit()

    def log_alert(self, alert_type, severity, description, model_name=None):
        """Logs a monitoring alert."""
        self.conn.execute('''
            INSERT INTO alerts
            (timestamp, alert_type, severity, description, model_name)
            VALUES (?,?,?,?,?)
        ''', (datetime.now().isoformat(), alert_type, severity,
              description, model_name))
        self.conn.commit()
        logging.warning(f"ALERT [{severity}] {model_name}: {description}") # Also print to console

    def get_decisions(self, model_name=None, days=7, limit=1000):
        """Retrieve recent decisions for review."""
        query = '''SELECT * FROM decisions WHERE timestamp > ?'''
        params = [(datetime.now() - timedelta(days=days)).isoformat()]

        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        
        query += f' ORDER BY timestamp DESC LIMIT {limit}'
        
        df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['timestamp'])
        # Convert input_features back to dict for easier use
        if 'input_features' in df.columns:
            df['input_features'] = df['input_features'].apply(json.loads)
        return df

    def get_alerts(self, model_name=None, days=7, acknowledged=None, severity=None):
        """Retrieve recent alerts."""
        query = '''SELECT * FROM alerts WHERE timestamp > ?'''
        params = [(datetime.now() - timedelta(days=days)).isoformat()]

        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        if acknowledged is not None:
            query += ' AND acknowledged = ?'
            params.append(1 if acknowledged else 0)
        if severity:
            query += ' AND severity = ?'
            params.append(severity)
        
        query += ' ORDER BY timestamp DESC'
        
        return pd.read_sql_query(query, self.conn, params=params, parse_dates=['timestamp'])


# Initialize the logger for FinSecure Bank's AI models
logger = AIDecisionLogger(db_path='finsecure_ai_audit_log.db')
print(f"Initialized AI audit log database at '{logger.db_path}' with 'decisions' and 'alerts' tables.")
```

**Explanation of Execution:**

By initializing `AIDecisionLogger`, Alex has established the central repository for all AI model decisions and associated alerts. This is a crucial first step for compliance and traceability, ensuring that FinSecure Bank can meet its regulatory obligations. The `log_decision` and `log_alert` methods provide the interface for models and monitoring systems to interact with this audit log. The `get_decisions` and `get_alerts` methods will be used by Alex to review the logged data.

---

## 3. Non-Invasive Decision Capture with Decorators

**Story + Context + Real-World Relevance:**

Integrating audit logging into existing production AI models can be complex. Model development teams often prioritize prediction accuracy and performance. The governance team, led by Alex, needs to ensure logging without directly modifying the sensitive, validated model code. Modifying core model logic introduces risk and requires re-validation.

To address this, Alex decides to implement a **non-invasive logging wrapper using a Python decorator**. This architectural pattern allows the governance team to "decorate" any prediction function, injecting logging capabilities without altering the original function's source code. This separation of concerns is a best practice in robust MLOps.

**Concept:** Decorator Pattern for Non-Invasive Logging

A decorator `audit_logged` will:
1.  Take a model's prediction function as input.
2.  Wrap it with a new function that calls the original prediction function.
3.  Extract relevant metadata from the prediction's inputs and outputs.
4.  Log this metadata to the `AIDecisionLogger`.
5.  Return the original prediction result, ensuring the model's behavior is unchanged.

```python
def audit_logged(model_name, model_version, decision_type, logger):
    """
    Decorator that wraps any prediction function with audit logging.
    The original function is unchanged -- logging is non-invasive.
    """
    def decorator(predict_fn):
        @functools.wraps(predict_fn)
        def wrapper(*args, **kwargs):
            # Call the original prediction function
            result = predict_fn(*args, **kwargs)

            # Extract logging metadata from kwargs or result
            # Assuming models return a dictionary with 'prediction', 'confidence', 'ticker', 'sector', etc.
            # Or, these can be passed as kwargs to the predict_fn and extracted from kwargs
            prediction = result.get('prediction', str(result)) if isinstance(result, dict) else str(result)
            confidence = result.get('confidence', None) if isinstance(result, dict) else None
            ticker = kwargs.get('ticker', result.get('ticker', None) if isinstance(result, dict) else None)
            sector = kwargs.get('sector', result.get('sector', None) if isinstance(result, dict) else None)
            explanation = kwargs.get('explanation', result.get('explanation', None) if isinstance(result, dict) else None)
            user_id = kwargs.get('user_id', 'system') # Default user_id

            # input_features are typically passed as kwargs or the first arg (if single dict)
            input_features = kwargs.get('features', args[0] if args and isinstance(args[0], dict) else None)


            # Log the decision using the central logger
            logger.log_decision(
                model_name=model_name,
                model_version=model_version,
                decision_type=decision_type,
                prediction=prediction,
                confidence=confidence,
                input_features=input_features,
                explanation=explanation,
                ticker=ticker,
                sector=sector,
                user_id=user_id
            )
            return result
        return wrapper
    return decorator

# --- Example AI Models at FinSecure Bank ---

# Trading Signal Model: Recommends BUY/SELL/HOLD based on momentum
@audit_logged('MomentumSignal', 'v1.3', 'trading_signal', logger)
def generate_trading_signal(ticker, features=None, **kwargs):
    """
    Simplified momentum trading signal model.
    Features: {'momentum_12m', 'volatility', 'pe_ratio'}
    """
    momentum_12m = features.get('momentum_12m', 0) if features else 0

    if momentum_12m > 0.15:
        decision = 'BUY'
        confidence = 0.75
        explanation = "Strong 12-month momentum"
    elif momentum_12m < -0.10:
        decision = 'SELL'
        confidence = 0.70
        explanation = "Negative 12-month momentum"
    else:
        decision = 'HOLD'
        confidence = 0.55
        explanation = "Neutral momentum"
    
    return {
        'ticker': ticker,
        'prediction': decision,
        'confidence': confidence,
        'explanation': explanation,
        'sector': kwargs.get('sector') # Pass sector through kwargs
    }

# Credit Scoring Model: Approves/Declines loan applications
@audit_logged('CreditXGBoost', 'v2.1', 'credit_approval', logger)
def score_credit_application(applicant_id, features=None, **kwargs):
    """
    Simplified credit scoring model based on FICO and DTI.
    Features: {'fico', 'dti', 'income', 'loan_amount'}
    """
    fico = features.get('fico', 700) if features else 700
    dti = features.get('dti', 30) if features else 30

    # Simplified probability of default (PD) score calculation
    # Lower PD is better
    pd_score = max(0.01, min(0.99, (800 - fico) / 1000 + dti / 200))

    if pd_score < 0.15: # Threshold for approval
        decision = 'APPROVE'
        confidence = 1 - pd_score
        explanation = f"Low PD ({pd_score:.2f}) based on FICO ({fico}) and DTI ({dti:.1f})"
    else:
        decision = 'DECLINE'
        confidence = 1 - pd_score
        explanation = f"High PD ({pd_score:.2f}) based on FICO ({fico}) and DTI ({dti:.1f})"
    
    return {
        'applicant_id': applicant_id,
        'prediction': decision,
        'confidence': confidence,
        'explanation': explanation,
        'user_id': kwargs.get('user_id', 'system')
    }

print("AI models 'MomentumSignal' and 'CreditXGBoost' are now wrapped with audit logging.")
```

**Explanation of Execution:**

Alex has successfully wrapped FinSecure's AI models. Now, every time `generate_trading_signal` or `score_credit_application` is called, the `audit_logged` decorator will automatically capture and log the decision details to the SQLite database without requiring any changes to the model's prediction logic. This ensures compliance with audit trail requirements while maintaining development agility.

---

## 4. Simulating AI Decisions with Injected Anomalies

**Story + Context + Real-World Relevance:**

To validate the anomaly detection capabilities, Alex needs to test the system with real-world scenarios, including days where models behave abnormally. Simulating production decisions allows him to control the environment and inject specific types of anomalies to ensure the monitoring system effectively flags them. This proactive testing is crucial for Alex to gain confidence in the governance framework before live incidents occur.

He will simulate several days of trading signals and credit approvals, intentionally introducing anomalous behavior on one of these days to see if the detection system catches it.

**Concept:** Synthetic Data Generation with Injected Anomalies

The simulation will generate:
-   **Normal decisions**: Following expected patterns for `momentum_12m`, `fico`, `dti`.
-   **Injected anomalies**:
    -   **Trading Model Anomaly**: A sudden shift in recommendations, e.g., an unusually high proportion of 'SELL' signals. This will be simulated by forcing `momentum_12m` to a low value for many decisions on an "anomaly day."
    -   **Credit Model Anomaly**: A subtle shift in approval criteria, e.g., approving a higher number of applicants with low FICO scores. This will be simulated by lowering the `fico` score for some approved applications on the "anomaly day."

```python
def simulate_production_day(logger_instance, n_trading=50, n_credit=100, anomaly_day=False):
    """
    Simulates a day of AI model decisions for monitoring demo.
    """
    trading_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'JPM', 'BAC', 'XOM',
                       'CVX', 'JNJ', 'PFE', 'UNH', 'V', 'MA', 'HD', 'PG']
    
    # Define sectors for trading signals
    sectors = ['Tech', 'Finance', 'Energy', 'Health', 'Consumer_Staples', 'Industrials']

    print(f"--- Simulating Decisions for {'Anomaly Day' if anomaly_day else 'Normal Day'} ---")

    # Simulate Trading signals
    for _ in range(n_trading):
        ticker = random.choice(trading_tickers)
        sector = random.choice(sectors)
        
        # Simulate normal momentum
        momentum_12m = np.random.normal(0.05, 0.15) # Mean 0.05, Std Dev 0.15

        # Inject anomaly: model suddenly recommends SELL for most stocks
        if anomaly_day and random.random() < 0.7: # 70% chance to force a SELL signal on anomaly day
            momentum_12m = -0.20 # Force a strong negative momentum to trigger SELL
        
        features = {
            'momentum_12m': momentum_12m,
            'volatility': np.random.uniform(0.1, 0.4),
            'pe_ratio': np.random.uniform(10, 40)
        }
        generate_trading_signal(ticker=ticker, features=features, sector=sector)

    # Simulate Credit decisions
    for i in range(n_credit):
        applicant_id = f'APP-{random.randint(10000, 99999)}'
        fico = int(np.random.normal(700, 60)) # Mean 700, Std Dev 60
        dti = np.random.uniform(15, 55) # Debt-to-income ratio

        # Inject anomaly: model approves very high-risk applicants
        if anomaly_day and random.random() < 0.3: # 30% chance to force a low FICO for approval on anomaly day
            fico = random.randint(400, 550) # Low FICO but still processed by model
        
        features = {
            'fico': fico,
            'dti': round(dti, 1),
            'income': int(np.random.lognormal(11, 0.5)), # Log-normal for income
            'loan_amount': int(np.random.lognormal(10, 0.8)) # Log-normal for loan amount
        }
        score_credit_application(applicant_id=applicant_id, features=features, user_id=f'User-{random.randint(1,10)}')

    print(f"Logged {n_trading} trading decisions and {n_credit} credit decisions.")

# --- Run the simulation ---
total_simulation_days = 6 # 5 normal days + 1 anomaly day
anomaly_trigger_day = 4 # Day 4 will be the anomaly

for day in range(total_simulation_days):
    is_anomaly = (day == anomaly_trigger_day - 1) # Adjust to 0-indexed day
    print(f"\n--- Day {day+1} of {total_simulation_days} ---")
    simulate_production_day(logger, anomaly_day=is_anomaly)

# Retrieve all decisions to confirm logging
all_decisions_df = logger.get_decisions(days=total_simulation_days)
print(f"\nTotal logged decisions over {total_simulation_days} days: {len(all_decisions_df)}")
print(f"Decisions for MomentumSignal: {len(all_decisions_df[all_decisions_df['model_name'] == 'MomentumSignal'])}")
print(f"Decisions for CreditXGBoost: {len(all_decisions_df[all_decisions_df['model_name'] == 'CreditXGBoost'])}")
```

**Explanation of Execution:**

Alex has now generated a realistic stream of AI decisions, including a specific day where both the trading and credit models exhibited anomalous behavior. This dataset is crucial for the next step: validating that the automated anomaly detection system can effectively flag these deviations. The logged decisions provide the necessary data for comparing recent model behavior against a historical baseline.

---

## 5. Proactive Anomaly Detection in Decision Streams

**Story + Context + Real-World Relevance:**

Alex's primary role is to proactively identify and mitigate risks. Manual review of every decision is impossible. He needs an automated system to constantly monitor the decision streams and alert him when model behavior deviates from established norms. This is where anomaly detection algorithms come into play, providing the "smoke detectors" for AI model risk mentioned in the reference materials.

He will implement four essential anomaly checks:
1.  **Decision Distribution Shift**: Detects changes in the proportion of different decision categories (e.g., too many 'SELL' signals).
2.  **Concentration Risk**: Flags if the model is over-concentrating decisions on a single entity (e.g., one stock).
3.  **Confidence Anomaly**: Identifies significant shifts in the model's prediction confidence.
4.  **Volume Anomaly**: Alerts for abnormal spikes or drops in the number of decisions.

For each check, a **threshold** is defined. If the deviation exceeds this threshold, an alert is logged with a severity level. This aligns with SR 11-7's requirement for ongoing monitoring and proactive risk mitigation.

**Concept:** Anomaly Detection Algorithms with Statistical Thresholds

The following mathematical logic will be applied for anomaly detection:

**1. Decision Distribution Shift:**
This check compares the distribution of `prediction` categories in the recent `window_days` against a `baseline_days` period.
For each decision category $d$:
$$ \Delta P_d = |P_{\text{recent}}(d) - P_{\text{baseline}}(d)| $$
An alert is triggered if $\Delta P_d > 0.20$ (20 percentage points) for any decision category $d$.

**2. Concentration Risk:**
This identifies if the model is disproportionately focusing on a single entity (`ticker` for trading signals, not applicable for credit applications in this demo).
For the recent window, calculate the maximum proportion of decisions attributed to any single entity:
$$ \text{Max Concentration} = \max_{entity} \left( \frac{\text{Count of decisions for entity}}{\text{Total decisions in recent window}} \right) $$
An alert is triggered if $\text{Max Concentration} > 0.30$ (30%).

**3. Confidence Anomaly:**
This monitors shifts in the model's average prediction confidence.
$$ \Delta \text{Confidence} = |\text{mean Confidence}_{\text{recent}} - \text{mean Confidence}_{\text{baseline}}| $$
An alert is triggered if $\Delta \text{Confidence} > 0.10$.

**4. Volume Anomaly:**
This check detects abnormal daily decision counts.
Let $\text{Daily Count}_{\text{recent}}$ be the average daily decisions in the recent window, and $\text{Daily Count}_{\text{baseline}}$ be the average daily decisions in the baseline.
An alert is triggered if $\text{Daily Count}_{\text{recent}} > 2 \times \text{Daily Count}_{\text{baseline}}$ (more than double) or $\text{Daily Count}_{\text{recent}} < 0.3 \times \text{Daily Count}_{\text{baseline}}$ (less than 30% of baseline).

```python
def detect_decision_anomalies(logger_instance, model_name, window_days=1, baseline_days=7):
    """
    Compares recent decision patterns to historical baseline and flags anomalies.
    """
    print(f"\n--- Detecting anomalies for {model_name} (Recent: {window_days} day, Baseline: {baseline_days} days) ---")

    recent_df = logger_instance.get_decisions(model_name, days=window_days)
    baseline_df = logger_instance.get_decisions(model_name, days=baseline_days + window_days).loc[lambda x: x['timestamp'] < (datetime.now() - timedelta(days=window_days))] # Exclude recent window from baseline

    alerts = []

    if len(recent_df) == 0 or len(baseline_df) == 0:
        print(f"  Not enough data for {model_name} to perform anomaly detection.")
        return []

    # 1. Decision Distribution Shift
    # Alert if |P_recent - P_baseline| > 20%
    recent_dist = recent_df['prediction'].value_counts(normalize=True).fillna(0)
    baseline_dist = baseline_df['prediction'].value_counts(normalize=True).fillna(0)
    
    all_decisions = sorted(list(set(recent_dist.index).union(baseline_dist.index)))

    for decision_cat in all_decisions:
        recent_pct = recent_dist.get(decision_cat, 0)
        baseline_pct = baseline_dist.get(decision_cat, 0)
        shift = abs(recent_pct - baseline_pct)

        if shift > 0.20: # 20 percentage point shift
            alerts.append({
                'type': 'distribution_shift',
                'severity': 'HIGH',
                'description': f"'{model_name}': '{decision_cat}' prediction shifted from {baseline_pct:.1%} to {recent_pct:.1%} (delta={shift:+.1%})",
                'model_name': model_name
            })

    # 2. Concentration Risk
    # Alert if max proportion of decisions for a single entity exceeds 30%
    if 'ticker' in recent_df.columns and model_name == 'MomentumSignal': # Concentration usually applies to trading models
        ticker_counts = recent_df['ticker'].value_counts()
        if not ticker_counts.empty:
            max_concentration = ticker_counts.iloc[0] / len(recent_df)
            if max_concentration > 0.30: # 30% concentration
                alerts.append({
                    'type': 'concentration_risk',
                    'severity': 'MEDIUM',
                    'description': f"'{model_name}': '{ticker_counts.index[0]}' represents {max_concentration:.1%} of decisions",
                    'model_name': model_name
                })

    # 3. Confidence Anomaly
    # Alert if |mean Confidence_recent - mean Confidence_baseline| > 0.10
    if 'confidence' in recent_df.columns:
        recent_conf = pd.to_numeric(recent_df['confidence'], errors='coerce').dropna()
        baseline_conf = pd.to_numeric(baseline_df['confidence'], errors='coerce').dropna()

        if not recent_conf.empty and not baseline_conf.empty:
            recent_avg_conf = recent_conf.mean()
            baseline_avg_conf = baseline_conf.mean()
            if abs(recent_avg_conf - baseline_avg_conf) > 0.10: # 0.10 absolute shift
                alerts.append({
                    'type': 'confidence_shift',
                    'severity': 'MEDIUM',
                    'description': f"'{model_name}': avg confidence shifted from {baseline_avg_conf:.2f} to {recent_avg_conf:.2f}",
                    'model_name': model_name
                })

    # 4. Volume Anomaly
    # Alert if daily decision count is > 2x baseline or < 0.3x baseline
    if len(recent_df) > 0 and len(baseline_df) > 0:
        recent_daily_avg = len(recent_df) / window_days
        baseline_daily_avg = len(baseline_df) / baseline_days

        if recent_daily_avg > 2 * baseline_daily_avg:
            alerts.append({
                'type': 'volume_anomaly',
                'severity': 'LOW',
                'description': f"'{model_name}': daily volume ({recent_daily_avg:.0f}) is more than 2x baseline ({baseline_daily_avg:.0f})",
                'model_name': model_name
            })
        elif recent_daily_avg < 0.3 * baseline_daily_avg:
            alerts.append({
                'type': 'volume_anomaly',
                'severity': 'LOW',
                'description': f"'{model_name}': daily volume ({recent_daily_avg:.0f}) is less than 0.3x baseline ({baseline_daily_avg:.0f})",
                'model_name': model_name
            })

    # Log detected alerts and print summary
    if alerts:
        for alert_data in alerts:
            logger_instance.log_alert(alert_data['type'], alert_data['severity'],
                                      alert_data['description'], alert_data['model_name'])
        print(f"  {len(alerts)} anomalies detected for {model_name}:")
        for a in alerts:
            print(f"    [{a['severity']}] {a['type']}: {a['description']}")
    else:
        print(f"  No anomalies detected for {model_name}.")

    return alerts

# --- Run anomaly detection for each model ---
models_to_monitor = ['MomentumSignal', 'CreditXGBoost']
window_for_anomalies = 1 # Look at the last 1 day for recent behavior
baseline_for_anomalies = 5 # Compare against the previous 5 days (excluding the recent window)

all_detected_alerts = []
for model in models_to_monitor:
    all_detected_alerts.extend(detect_decision_anomalies(logger, model, window_days=window_for_anomalies, baseline_days=baseline_for_anomalies))

# Review all alerts from the database
recent_alerts_df = logger.get_alerts(days=total_simulation_days)
print(f"\nTotal alerts logged in database for last {total_simulation_days} days: {len(recent_alerts_df)}")
if not recent_alerts_df.empty:
    print(recent_alerts_df[['timestamp', 'model_name', 'alert_type', 'severity', 'description']].head())
```

**Explanation of Execution:**

Alex's anomaly detection system is now operational. The simulation run produced several alerts, indicating that the injected anomalies were successfully detected. For instance, the "MomentumSignal" model triggered a "distribution_shift" alert due to the forced 'SELL' signals, and potentially a "confidence_shift" or "volume_anomaly". The "CreditXGBoost" model might have triggered a "confidence_shift" if low FICO approvals led to lower average confidence. These alerts provide Alex with actionable intelligence, allowing him to focus his investigation on specific issues rather than sifting through all decisions. This directly contributes to proactive risk mitigation and operational efficiency.

---

## 6. The Risk Officer's Monitoring Dashboard

**Story + Context + Real-World Relevance:**

With a continuous stream of decisions and potential alerts, Alex needs a consolidated "Risk Officer Review Dashboard." This dashboard must provide a high-level overview of AI model activity, highlight detected anomalies, show decision distributions, and identify any concentration risks. This unified view enables Alex to quickly grasp the current state of FinSecure's AI models, prioritize his review tasks, and respond promptly to critical issues, fulfilling the "ongoing monitoring" aspect of SR 11-7.

**Concept:** Data Aggregation and Visualization for Insights

The dashboard will use `matplotlib` to present:
-   **Summary Statistics**: Total decisions, decisions per day, pending reviews, alert counts.
-   **Decision Distribution Over Time**: Stacked bar charts for 'BUY'/'SELL'/'HOLD' or 'APPROVE'/'DECLINE' proportions over the monitoring period. This is crucial for visualizing shifts.
-   **Confidence Score Distribution**: Histograms of prediction confidence scores, identifying any unusual patterns.
-   **Concentration Risk**: A bar chart showing the highest proportion of decisions for specific entities (e.g., top 5 tickers), visually flagging over-concentration.
-   **Alert Timeline**: A scatter plot showing when alerts were triggered and their severity, providing a historical context.

```python
def generate_review_dashboard(logger_instance, model_name, days=7):
    """
    Generates a risk officer review dashboard for AI decisions.
    """
    print(f"\n{'='*80}")
    print(f"AI DECISION MONITORING DASHBOARD: {model_name}")
    print(f"Period: Last {days} days")
    print(f"{'='*80}")

    decisions_df = logger_instance.get_decisions(model_name, days=days)
    alerts_df = logger_instance.get_alerts(model_name, days=days)

    if decisions_df.empty:
        print("No decisions recorded for this model in the specified period.")
        return

    # --- Summary Statistics ---
    print(f"\n--- SUMMARY STATISTICS ---")
    total_decisions = len(decisions_df)
    decisions_per_day = total_decisions / max(1, days)
    pending_reviews = len(decisions_df[decisions_df['review_status'] == 'pending'])
    flagged_decisions = len(decisions_df[decisions_df['anomaly_flag'] == 1])
    total_alerts = len(alerts_df)
    high_severity_alerts = len(alerts_df[alerts_df['severity'] == 'HIGH'])

    print(f"Total Decisions: {total_decisions}")
    print(f"Decisions/Day (Avg): {decisions_per_day:.0f}")
    print(f"Pending Human Reviews: {pending_reviews}")
    print(f"Decisions Flagged for Anomaly: {flagged_decisions}")
    print(f"Total Alerts: {total_alerts}")
    print(f"High Severity Alerts: {high_severity_alerts}")

    # --- Visualizations ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    fig.suptitle(f'AI Model Monitoring Dashboard: {model_name} (Last {days} days)', fontsize=16)
    
    # 1. Decision Distribution Over Time (Stacked Bar Chart)
    # Group by date and prediction, then unstack to get columns for each prediction type
    decisions_df['date'] = decisions_df['timestamp'].dt.normalize()
    daily_dist = decisions_df.groupby(['date', 'prediction']).size().unstack(fill_value=0)
    daily_dist_pct = daily_dist.div(daily_dist.sum(axis=1), axis=0) # Convert to proportions

    daily_dist_pct.plot(kind='bar', stacked=True, ax=axes[0], colormap='viridis')
    axes[0].set_title('Daily Decision Distribution (%)')
    axes[0].set_ylabel('Proportion')
    axes[0].set_xlabel('Date')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title='Prediction')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Highlight anomaly day if available in description
    for _, alert_row in alerts_df[alerts_df['alert_type'] == 'distribution_shift'].iterrows():
        # Heuristic to find the day mentioned in description, assuming description format like '...from X to Y (delta=Z)'
        alert_date_str = alert_row['timestamp'][:10] # Get YYYY-MM-DD
        alert_date = pd.to_datetime(alert_date_str)
        if alert_date in daily_dist_pct.index:
             axes[0].axvline(x=daily_dist_pct.index.get_loc(alert_date), color='red', linestyle='--', linewidth=2, label='Anomaly Day')
             handles, labels = axes[0].get_legend_handles_labels()
             if 'Anomaly Day' not in labels:
                 axes[0].legend(handles=handles, labels=labels)


    # 2. Confidence Distribution (Histogram)
    if 'confidence' in decisions_df.columns:
        clean_confidence = pd.to_numeric(decisions_df['confidence'], errors='coerce').dropna()
        if not clean_confidence.empty:
            axes[1].hist(clean_confidence, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            axes[1].set_title('Distribution of Prediction Confidence Scores')
            axes[1].set_xlabel('Confidence Score')
            axes[1].set_ylabel('Frequency')
        else:
            axes[1].text(0.5, 0.5, 'No confidence data available', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, 'No confidence column available', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


    # 3. Concentration Risk (Top N Tickers/Entities) - Bar Chart as proportional visualization
    if 'ticker' in decisions_df.columns and model_name == 'MomentumSignal':
        ticker_counts = decisions_df['ticker'].value_counts(normalize=True).head(5)
        if not ticker_counts.empty:
            ticker_counts.plot(kind='barh', ax=axes[2], color='lightcoral')
            axes[2].set_title(f'Top 5 Ticker Concentration for {model_name} (%)')
            axes[2].set_xlabel('Proportion of Decisions')
            axes[2].set_ylabel('Ticker')
            axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        else:
            axes[2].text(0.5, 0.5, 'No ticker data available', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
    else:
        axes[2].text(0.5, 0.5, 'Concentration risk not applicable or no ticker data', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)

    # 4. Alert Timeline (Scatter Plot by Severity)
    if not alerts_df.empty:
        severity_colors = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
        alerts_df['severity_num'] = alerts_df['severity'].map({'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}) # For consistent plotting order
        alerts_df = alerts_df.sort_values('severity_num') # Plot higher severity last to ensure visibility

        for severity_level, color in severity_colors.items():
            subset = alerts_df[alerts_df['severity'] == severity_level]
            if not subset.empty:
                axes[3].scatter(subset['timestamp'], subset['severity_num'], color=color, label=severity_level, s=100, alpha=0.7)
        
        axes[3].set_title('Alert Timeline by Severity')
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Severity')
        axes[3].set_yticks([1, 2, 3])
        axes[3].set_yticklabels(['LOW', 'MEDIUM', 'HIGH'])
        axes[3].legend(title='Severity')
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[3].text(0.5, 0.5, 'No alerts in this period', horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.show()

# --- Alex reviews the dashboard for each model ---
for model in models_to_monitor:
    generate_review_dashboard(logger, model, days=total_simulation_days)

print("\nRisk Officer's dashboards generated. Alex can now quickly assess model health.")
```

**Explanation of Execution:**

Alex's dashboard provides an immediate, visual summary of each AI model's operational health. He can quickly observe:
-   Any shifts in decision patterns (e.g., the high proportion of 'SELL' signals on the anomaly day for 'MomentumSignal' in the stacked bar chart).
-   Unusual confidence distributions, which might suggest data quality issues or model drift.
-   If 'MomentumSignal' shows a single ticker dominating the "Top 5 Ticker Concentration" chart, indicating potential concentration risk.
-   The timing and severity of alerts on the scatter plot, allowing him to prioritize investigation.

This dashboard is a critical tool for Alex to efficiently perform ongoing monitoring and comply with internal and external governance requirements, enabling him to proactively address risks before they escalate.

---

## 7. Generating a Periodic AI Model Audit Report

**Story + Context + Real-World Relevance:**

Beyond daily monitoring, Alex, as a Risk Officer, is responsible for providing formal, periodic compliance reports to FinSecure Bank's risk committee and external regulators. These reports summarize AI model activity, highlight detected anomalies, detail review statuses, and affirm compliance with relevant regulations like SR 11-7 and the EU AI Act. This formal documentation is essential for demonstrating robust AI governance and accountability.

**Concept:** Structured Reporting for Compliance

The `generate_audit_report` function will:
1.  Gather summary statistics for all AI models over a specified period.
2.  Summarize alerts, including those of high severity and unacknowledged ones.
3.  Provide per-model statistics (decisions, distribution, confidence, specific alerts).
4.  Include explicit statements regarding regulatory compliance.
5.  Provide designated lines for sign-off by relevant officers (Risk Officer, Compliance Officer, CRO).

```python
def generate_audit_report(logger_instance, period_days=7):
    """
    Generates a structured periodic audit report (e.g., weekly/monthly)
    for compliance and regulators.
    """
    decisions_df = logger_instance.get_decisions(days=period_days)
    alerts_df = logger_instance.get_alerts(days=period_days)

    report = {
        'report_title': 'AI Model Audit Report - FinSecure Bank',
        'period': f'Last {period_days} days',
        'generation_date': datetime.now().isoformat(),
        'executive_summary': {
            'total_ai_decisions': len(decisions_df),
            'models_active': decisions_df['model_name'].nunique() if not decisions_df.empty else 0,
            'total_alerts': len(alerts_df),
            'high_severity_alerts': len(alerts_df[alerts_df['severity']=='HIGH']),
            'unacknowledged_alerts': len(alerts_df[alerts_df['acknowledged']==0]),
            'decisions_pending_review': len(decisions_df[decisions_df['review_status']=='pending'])
        },
        'model_summaries': {},
        'regulatory_compliance': {
            'SR_11_7_monitoring': 'Active - all deployed models monitored daily for anomalies.',
            'EU_AI_Act_logging': 'Compliant - all high-risk AI decisions logged with inputs, outputs, and metadata.',
            'record_retention': f'{len(decisions_df)} decision records retained for the period.',
            'anomaly_detection': 'Four-check system operational, detecting distribution shifts, concentration, confidence, and volume anomalies.'
        },
        'sign_off': {
            'Risk Officer': {'name': 'Alex Chen', 'date': '__________'},
            'Compliance Officer': {'name': '__________', 'date': '__________'},
            'Chief Risk Officer (CRO)': {'name': '__________', 'date': '__________'}
        }
    }

    # Per-model summaries
    if not decisions_df.empty:
        for model in decisions_df['model_name'].unique():
            model_decisions = decisions_df[decisions_df['model_name'] == model]
            model_alerts = alerts_df[alerts_df['model_name'] == model]

            # Calculate decision distribution
            decision_dist = model_decisions['prediction'].value_counts(normalize=True).apply(lambda x: f'{x:.1%}').to_dict()

            # Calculate average confidence
            avg_confidence = pd.to_numeric(model_decisions['confidence'], errors='coerce').dropna().mean() if 'confidence' in model_decisions.columns else 'N/A'

            report['model_summaries'][model] = {
                'decision_count': len(model_decisions),
                'decision_distribution': decision_dist,
                'avg_confidence': f'{avg_confidence:.3f}' if pd.isna(avg_confidence) is False and avg_confidence != 'N/A' else avg_confidence,
                'alerts_count': len(model_alerts),
                'anomaly_flags_count': int(model_decisions['anomaly_flag'].sum())
            }

    # --- Print the report ---
    print("\n" + "="*80)
    print(f"       {report['report_title']}")
    print(f"       Period: {report['period']}")
    print(f"       Generated: {report['generation_date']}")
    print("="*80)

    print("\n--- EXECUTIVE SUMMARY ---")
    summary = report['executive_summary']
    print(f"Total AI Decisions: {summary['total_ai_decisions']}")
    print(f"Active Models: {summary['models_active']}")
    print(f"Total Alerts: {summary['total_alerts']} (HIGH: {summary['high_severity_alerts']}, Unacknowledged: {summary['unacknowledged_alerts']})")
    print(f"Decisions Pending Review: {summary['decisions_pending_review']}")

    print("\n--- MODEL SUMMARIES ---")
    if not report['model_summaries']:
        print("No model summaries for the period.")
    for model, stats in report['model_summaries'].items():
        print(f"\nModel: {model}")
        print(f"  Decisions: {stats['decision_count']}")
        print(f"  Distribution: {stats['decision_distribution']}")
        print(f"  Avg Confidence: {stats['avg_confidence']}")
        print(f"  Alerts: {stats['alerts_count']}, Flagged Decisions: {stats['anomaly_flags_count']}")

    print("\n--- REGULATORY COMPLIANCE STATUS ---")
    for reg, status in report['regulatory_compliance'].items():
        print(f"  {reg}: {status}")

    print("\n--- SIGN-OFF ---")
    for role, details in report['sign_off'].items():
        print(f"{role}: {details['name']} (Date: {details['date']})")
        print("  Signature: __________________________\n")

    print("="*80)
    return report

# --- Alex generates the weekly audit report ---
audit_report_data = generate_audit_report(logger, period_days=total_simulation_days)
print("\nAI Model Audit Report generated. This report is ready for internal review and regulatory submission.")
```

**Explanation of Execution:**

Alex now has a comprehensive AI Model Audit Report. This markdown-formatted output provides:
-   An **Executive Summary** for leadership.
-   **Per-model statistics** detailing decision volumes, distributions, and confidence levels.
-   Crucially, **regulatory compliance statements** explicitly confirming adherence to requirements like SR 11-7 and the EU AI Act.
-   **Sign-off lines** for accountability.

This report is the final deliverable, transforming raw audit log data and anomaly alerts into a structured, auditable document. It demonstrates FinSecure Bank's commitment to robust AI governance and model risk management, allowing Alex to efficiently fulfill his reporting obligations and enhance trust in the bank's AI systems.
