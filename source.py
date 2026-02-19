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

class AIDecisionLogger:
    """
    Comprehensive audit logger for AI model decisions and monitoring alerts.
    Captures inputs, outputs, metadata, and enables review.
    """
    def __init__(self, db_path='ai_audit_log.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES, check_same_thread=False)
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

# Initialize a global logger instance. This is used by the decorators
# for AI models, allowing them to log decisions automatically.
_global_decision_logger = AIDecisionLogger(db_path='finsecure_ai_audit_log.db')
# Create an alias for compatibility with app.py imports
logger = _global_decision_logger
print(f"Initialized AI audit log database at '{_global_decision_logger.db_path}' with 'decisions' and 'alerts' tables.")

def audit_logged(model_name, model_version, decision_type, logger_instance):
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
            prediction = result.get('prediction', str(result)) if isinstance(result, dict) else str(result)
            confidence = result.get('confidence', None) if isinstance(result, dict) else None
            ticker = kwargs.get('ticker', result.get('ticker', None) if isinstance(result, dict) else None)
            sector = kwargs.get('sector', result.get('sector', None) if isinstance(result, dict) else None)
            explanation = kwargs.get('explanation', result.get('explanation', None) if isinstance(result, dict) else None)
            user_id = kwargs.get('user_id', 'system') # Default user_id

            # input_features are typically passed as kwargs or the first arg (if single dict)
            input_features = kwargs.get('features', args[0] if args and isinstance(args[0], dict) else None)

            # Log the decision using the central logger instance provided to the decorator
            logger_instance.log_decision(
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
# These models use the _global_decision_logger instance via the decorator
@audit_logged('MomentumSignal', 'v1.3', 'trading_signal', _global_decision_logger)
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
        'sector': kwargs.get('sector')
    }

@audit_logged('CreditXGBoost', 'v2.1', 'credit_approval', _global_decision_logger)
def score_credit_application(applicant_id, features=None, **kwargs):
    """
    Simplified credit scoring model based on FICO and DTI.
    Features: {'fico', 'dti', 'income', 'loan_amount'}
    """
    fico = features.get('fico', 700) if features else 700
    dti = features.get('dti', 30) if features else 30

    # Simplified probability of default (PD) score calculation
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

def simulate_production_day(logger_instance: AIDecisionLogger, n_trading: int = 50, n_credit: int = 100, anomaly_day: bool = False):
    """
    Simulates a day of AI model decisions for monitoring demo.
    """
    trading_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'JPM', 'BAC', 'XOM',
                       'CVX', 'JNJ', 'PFE', 'UNH', 'V', 'MA', 'HD', 'PG']
    sectors = ['Tech', 'Finance', 'Energy', 'Health', 'Consumer_Staples', 'Industrials']

    print(f"--- Simulating Decisions for {'Anomaly Day' if anomaly_day else 'Normal Day'} ---")

    # Simulate Trading signals
    for _ in range(n_trading):
        ticker = random.choice(trading_tickers)
        sector = random.choice(sectors)
        momentum_12m = np.random.normal(0.05, 0.15) # Mean 0.05, Std Dev 0.15

        if anomaly_day and random.random() < 0.7: # 70% chance to force a SELL signal on anomaly day
            momentum_12m = -0.20 # Force a strong negative momentum to trigger SELL

        features = {
              'momentum_12m': momentum_12m,
            'volatility': np.random.uniform(0.1, 0.4),
            'pe_ratio': np.random.uniform(10, 40)
        }
        # Call the decorated function, which will use _global_decision_logger
        generate_trading_signal(ticker=ticker, features=features, sector=sector)

    # Simulate Credit decisions
    for i in range(n_credit):
        applicant_id = f'APP-{random.randint(10000, 99999)}'
        fico = int(np.random.normal(700, 60)) # Mean 700, Std Dev 60
        dti = np.random.uniform(15, 55) # Debt-to-income ratio

        if anomaly_day and random.random() < 0.3: # 30% chance to force a low FICO for approval on anomaly day
            fico = random.randint(400, 550) # Low FICO but still processed by model

        features = {
              'fico': fico,
            'dti': round(dti, 1),
            'income': int(np.random.lognormal(11, 0.5)), # Log-normal for income
            'loan_amount': int(np.random.lognormal(10, 0.8)) # Log-normal for loan amount
        }
        # Call the decorated function, which will use _global_decision_logger
        score_credit_application(applicant_id=applicant_id, features=features, user_id=f'User-{random.randint(1,10)}')

    print(f"Logged {n_trading} trading decisions and {n_credit} credit decisions.")


def detect_decision_anomalies(logger_instance: AIDecisionLogger, model_name: str, window_days: int = 1, baseline_days: int = 7) -> list:
    """
    Compares recent decision patterns to historical baseline and flags anomalies.
    Logs detected anomalies as alerts.
    """
    print(f"\n--- Detecting anomalies for {model_name} (Recent: {window_days} day, Baseline: {baseline_days} days) ---")

    recent_df = logger_instance.get_decisions(model_name, days=window_days)
    # Exclude the recent window from the baseline period to avoid overlap
    baseline_df = logger_instance.get_decisions(model_name, days=baseline_days + window_days)
    baseline_df = baseline_df.loc[baseline_df['timestamp'] < (datetime.now() - timedelta(days=window_days))]

    alerts = []

    if len(recent_df) == 0 or len(baseline_df) == 0:
        print(f"  Not enough data for {model_name} to perform anomaly detection.")
        return []

    # 1. Decision Distribution Shift
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

    # 2. Concentration Risk (applies mostly to trading models)
    if 'ticker' in recent_df.columns and model_name == 'MomentumSignal':
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

def generate_review_dashboard(logger_instance: AIDecisionLogger, model_name: str, days: int = 7, save_path: str = None):
    """
    Generates a risk officer review dashboard for AI decisions and optionally saves it.
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
    # Note: 'anomaly_flag' is not currently set by the anomaly detection functions in this code.
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

    # 1. Daily Decision Distribution (%)
    decisions_df['date'] = decisions_df['timestamp'].dt.normalize()
    daily_dist = decisions_df.groupby(['date', 'prediction']).size().unstack(fill_value=0)
    daily_dist_pct = daily_dist.div(daily_dist.sum(axis=1), axis=0)

    daily_dist_pct.plot(kind='bar', stacked=True, ax=axes[0], colormap='viridis')
    axes[0].set_title('Daily Decision Distribution (%)')
    axes[0].set_ylabel('Proportion')
    axes[0].set_xlabel('Date')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Highlight anomaly day if a 'distribution_shift' alert exists for that day
    for _, alert_row in alerts_df[alerts_df['alert_type'] == 'distribution_shift'].iterrows():
        alert_date = pd.to_datetime(alert_row['timestamp']).normalize()
        if alert_date in daily_dist_pct.index:
            try:
                date_index_pos = daily_dist_pct.index.get_loc(alert_date)
                # Only add label once
                if 'Anomaly Day' not in [l.get_label() for l in axes[0].lines]:
                    axes[0].axvline(x=date_index_pos, color='red', linestyle='--', linewidth=2, label='Anomaly Day')
                else:
                    axes[0].axvline(x=date_index_pos, color='red', linestyle='--', linewidth=2)
            except KeyError:
                pass # Date might not be in the current daily_dist_pct index

    # Update legend to include 'Anomaly Day' only once
    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    axes[0].legend(unique_labels.values(), unique_labels.keys(), title='Prediction / Anomaly')


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


    # 3. Concentration Risk (Top N Tickers/Entities)
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
        alerts_df['severity_num'] = alerts_df['severity'].map({'LOW': 1, 'MEDIUM': 2, 'HIGH': 3})
        alerts_df = alerts_df.sort_values('severity_num')

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

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
        print(f"Dashboard saved to {save_path}")
    # plt.show() # Commented out for library usage; app.py can decide to show or save


def generate_audit_report(logger_instance: AIDecisionLogger, period_days: int = 7) -> dict:
    """
    Generates a structured periodic audit report (e.g., weekly/monthly)
    for compliance and regulators. Returns the report as a dictionary.
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

            decision_dist = model_decisions['prediction'].value_counts(normalize=True).apply(lambda x: f'{x:.1%}').to_dict()
            avg_confidence = pd.to_numeric(model_decisions['confidence'], errors='coerce').dropna().mean()
            avg_confidence_str = f'{avg_confidence:.3f}' if pd.isna(avg_confidence) is False else 'N/A'

            report['model_summaries'][model] = {
                  'decision_count': len(model_decisions),
                'decision_distribution': decision_dist,
                'avg_confidence': avg_confidence_str,
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

def run_finsecure_analytics_pipeline(
    logger_instance: AIDecisionLogger,
    total_simulation_days: int = 6,
    anomaly_trigger_day: int = 4, # Day 4 will be the anomaly (1-indexed)
    n_trading_decisions_per_day: int = 50,
    n_credit_decisions_per_day: int = 100,
    window_for_anomalies_days: int = 1,
    baseline_for_anomalies_days: int = 5,
    models_to_monitor: list = None,
    dashboard_save_dir: str = None
) -> dict:
    """
    Orchestrates the simulation of AI decisions, anomaly detection,
    dashboard generation, and audit report generation for FinSecure Bank.

    Args:
        logger_instance: An initialized AIDecisionLogger instance.
        total_simulation_days: The total number of days to simulate.
        anomaly_trigger_day: The 1-indexed day when an anomaly should be triggered.
        n_trading_decisions_per_day: Number of trading decisions to simulate per day.
        n_credit_decisions_per_day: Number of credit decisions to simulate per day.
        window_for_anomalies_days: How many recent days to look at for anomalies.
        baseline_for_anomalies_days: How many previous days to use as a baseline for anomalies.
        models_to_monitor: A list of model names to apply anomaly detection and dashboards for.
                           Defaults to ['MomentumSignal', 'CreditXGBoost'].
        dashboard_save_dir: Directory to save generated dashboard plots. If None, plots are not saved.

    Returns:
        A dictionary containing the generated audit report data.
    """
    if models_to_monitor is None:
        models_to_monitor = ['MomentumSignal', 'CreditXGBoost']

    print(f"\n--- Starting FinSecure AI Analytics Pipeline ---")
    print(f"Total simulation days: {total_simulation_days}, Anomaly on day: {anomaly_trigger_day}")

    # --- Run the simulation ---
    for day_idx in range(total_simulation_days):
        is_anomaly = (day_idx == anomaly_trigger_day - 1) # Adjust to 0-indexed day
        print(f"\n--- Day {day_idx+1} of {total_simulation_days} Simulation ---")
        simulate_production_day(logger_instance,
                                n_trading=n_trading_decisions_per_day,
                                n_credit=n_credit_decisions_per_day,
                                anomaly_day=is_anomaly)

    # Retrieve all decisions to confirm logging
    all_decisions_df = logger_instance.get_decisions(days=total_simulation_days)
    print(f"\nTotal logged decisions over {total_simulation_days} days: {len(all_decisions_df)}")
    if not all_decisions_df.empty:
        print(f"Decisions for MomentumSignal: {len(all_decisions_df[all_decisions_df['model_name'] == 'MomentumSignal'])}")
        print(f"Decisions for CreditXGBoost: {len(all_decisions_df[all_decisions_df['model_name'] == 'CreditXGBoost'])}")
    else:
        print("No decisions logged during simulation.")


    # --- Run anomaly detection for each model ---
    all_detected_alerts = []
    print(f"\n--- Running Anomaly Detection ---")
    for model in models_to_monitor:
          all_detected_alerts.extend(detect_decision_anomalies(logger_instance, model,
                                                                 window_days=window_for_anomalies_days,
                                                                 baseline_days=baseline_for_anomalies_days))

    # Review all alerts from the database
    recent_alerts_df = logger_instance.get_alerts(days=total_simulation_days)
    print(f"\nTotal alerts logged in database for last {total_simulation_days} days: {len(recent_alerts_df)}")
    if not recent_alerts_df.empty:
          print(recent_alerts_df[['timestamp', 'model_name', 'alert_type', 'severity', 'description']].head())


    # --- Generate review dashboards for each model ---
    print(f"\n--- Generating Review Dashboards ---")
    for model in models_to_monitor:
        save_path = None
        if dashboard_save_dir:
            import os
            os.makedirs(dashboard_save_dir, exist_ok=True)
            save_path = os.path.join(dashboard_save_dir, f'dashboard_{model}_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
        generate_review_dashboard(logger_instance, model, days=total_simulation_days, save_path=save_path)
    plt.close('all') # Close all plots generated by the dashboard function to free memory


    # --- Generate the weekly audit report ---
    print(f"\n--- Generating AI Model Audit Report ---")
    audit_report_data = generate_audit_report(logger_instance, period_days=total_simulation_days)
    print("\nAI Model Audit Report generated. This report is ready for internal review and regulatory submission.")
    print(f"\n--- FinSecure AI Analytics Pipeline Completed ---")
    return audit_report_data

if __name__ == '__main__':
    # This block ensures the pipeline runs only when the script is executed directly.
    # It uses the _global_decision_logger initialized at the module level.
    run_finsecure_analytics_pipeline(
        _global_decision_logger,
        total_simulation_days=6,
        anomaly_trigger_day=4,
        n_trading_decisions_per_day=50,
        n_credit_decisions_per_day=100,
        window_for_anomalies_days=1,
        baseline_for_anomalies_days=5,
        models_to_monitor=['MomentumSignal', 'CreditXGBoost'],
        dashboard_save_dir='./dashboards' # Example: save dashboards to a 'dashboards' folder
    )
