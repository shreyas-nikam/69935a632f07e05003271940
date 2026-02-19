import os
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from source import *  # AIDecisionLogger + simulation + detection + reporting

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(page_title="QuLab: Lab 39: Audit Logging Demo", layout="wide")

# =============================================================================
# Pedagogy-first UI helpers (non-technical language)
# =============================================================================
def callout(title: str, body: str, kind: str = "info"):
    if kind == "info":
        st.info(f"**{title}**\n\n{body}")
    elif kind == "warning":
        st.warning(f"**{title}**\n\n{body}")
    elif kind == "success":
        st.success(f"**{title}**\n\n{body}")
    else:
        st.write(f"**{title}**\n\n{body}")

def assumptions_box(lines):
    st.caption("**Assumptions (for interpretation)**")
    st.markdown("\n".join([f"- {x}" for x in lines]))

def evidence_box(lines):
    st.caption("**Evidence / Documentation you should expect**")
    st.markdown("\n".join([f"- {x}" for x in lines]))

def pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{100*x:.1f}%"

def start_here_path():
    st.markdown("### Start Here: A governance-first learning path")
    st.markdown(
        """
1) **Generate a decision stream** (like production throughput)  
2) **Run transparent monitoring checks** and write alerts to the audit log  
3) **Triage the dashboard** like a Risk Officer (what changed? what needs review?)  
4) **Produce a committee-ready report** (evidence + sign-off)
        """.strip()
    )

def checkpoint_question(question: str, options: list, correct_option: str, explanation: str):
    st.markdown("#### Checkpoint (test your intuition)")
    ans = st.radio(question, options, index=None)
    if ans is None:
        st.caption("Pick one option to see the explanation.")
        return
    if ans == correct_option:
        st.success(f"Correct. {explanation}")
    else:
        st.info(f"Not quite. {explanation}")

def compute_logging_coverage(decisions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Coverage = % non-null for key governance fields.
    This is a *learning* metric: if coverage is low, the log is not defensible.
    """
    if decisions_df is None or decisions_df.empty:
        return pd.DataFrame(columns=["Field", "Coverage"])
    key_fields = [
        "timestamp",
        "model_name",
        "model_version",
        "input_hash",
        "input_features",
        "prediction",
        "confidence",
        "explanation",
        "portfolio_id",
        "user_id",
        "review_status",
        "review_notes",
        "anomaly_flag",
        "anomaly_reason",
    ]
    present = [c for c in key_fields if c in decisions_df.columns]
    rows = [{"Field": c, "Coverage": decisions_df[c].notna().mean()} for c in present]
    return pd.DataFrame(rows).sort_values("Coverage")

# =============================================================================
# Session state initialization
# =============================================================================
DB_PATH = "finsecure_ai_audit_log.db"

def initialize_session_state():
    if "logger" not in st.session_state:
        st.session_state["logger"] = AIDecisionLogger(db_path=DB_PATH)
        st.session_state.logger._create_tables()

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    # Simulation defaults (chosen to demonstrate the learning narrative)
    st.session_state.setdefault("total_simulation_days", 6)
    st.session_state.setdefault("anomaly_trigger_day", 4)  # 1-indexed for UI
    st.session_state.setdefault("n_trading_decisions_per_day", 50)
    st.session_state.setdefault("n_credit_decisions_per_day", 100)

    # Monitoring defaults
    st.session_state.setdefault("recent_window_days", 2)
    st.session_state.setdefault("baseline_window_days", 4)

    # Dashboard controls
    st.session_state.setdefault("selected_dashboard_model", None)
    st.session_state.setdefault("dashboard_review_period", 7)

    # Report controls
    st.session_state.setdefault("audit_report_period", 14)

    # Cached dataframes
    st.session_state.setdefault("current_decisions_df", pd.DataFrame())
    st.session_state.setdefault("current_alerts_df", pd.DataFrame())

initialize_session_state()

# =============================================================================
# Data access helpers (pull via SQL, because the logger exposes model-filtered helpers)
# =============================================================================
def get_decisions_df() -> pd.DataFrame:
    if st.session_state.current_decisions_df is None or st.session_state.current_decisions_df.empty:
        try:
            df = pd.read_sql_query("SELECT * FROM decisions", st.session_state.logger.conn)
            st.session_state.current_decisions_df = df
        except Exception:
            st.session_state.current_decisions_df = pd.DataFrame()
    return st.session_state.current_decisions_df

def get_alerts_df() -> pd.DataFrame:
    if st.session_state.current_alerts_df is None or st.session_state.current_alerts_df.empty:
        try:
            df = pd.read_sql_query("SELECT * FROM alerts", st.session_state.logger.conn)
            st.session_state.current_alerts_df = df
        except Exception:
            st.session_state.current_alerts_df = pd.DataFrame()
    return st.session_state.current_alerts_df

def invalidate_caches():
    st.session_state.current_decisions_df = pd.DataFrame()
    st.session_state.current_alerts_df = pd.DataFrame()

# =============================================================================
# Actions (non-technical descriptions; the work happens under the hood)
# =============================================================================
def reset_demo():
    """Clear the demo database so the learning path can be rerun cleanly."""
    try:
        # Close current connection
        try:
            st.session_state.logger.conn.close()
        except Exception:
            pass

        # Remove db file if present
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        # Recreate logger + tables
        st.session_state["logger"] = AIDecisionLogger(db_path=DB_PATH)
        st.session_state.logger._create_tables()

        invalidate_caches()
        st.success("Demo reset: audit log cleared and ready for a fresh run.")
    except Exception as e:
        st.error(f"Reset failed: {e}")

def acknowledge_alert(alert_id: int, acknowledged_by: str):
    """Mark an alert as triaged (ownership taken)."""
    try:
        st.session_state.logger.conn.execute(
            "UPDATE alerts SET acknowledged = 1, acknowledged_by = ? WHERE id = ?",
            (acknowledged_by, alert_id),
        )
        st.session_state.logger.conn.commit()
        invalidate_caches()
        st.success(f"Alert {alert_id} acknowledged by {acknowledged_by}.")
    except Exception as e:
        st.error(f"Error acknowledging alert {alert_id}: {e}")

# =============================================================================
# Sidebar + Navigation
# =============================================================================
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.sidebar.title("AI Model Governance Platform")
st.sidebar.caption("Audience: CFA charterholders • PMs • risk & research")

nav_options = [
    "Home",
    "Simulation & Data Generation",
    "Anomaly Detection",
    "Risk Officer Dashboard",
    "Audit Report",
]

try:
    current_index = nav_options.index(st.session_state.get("page", "Home"))
except ValueError:
    current_index = 0

selected_page = st.sidebar.selectbox("Choose a section", nav_options, index=current_index)
if selected_page != st.session_state["page"]:
    st.session_state["page"] = selected_page
    st.rerun()

# =============================================================================
# Global header
# =============================================================================
st.title("QuLab: Lab 39 — Audit Logging Demo")
st.divider()

# =============================================================================
# PAGE 1: HOME
# =============================================================================
if st.session_state["page"] == "Home":
    st.header("Audit Logging for AI Decisions: Reconstruct, Monitor, and Defend Model Actions")

    start_here_path()

    colA, colB = st.columns([3, 2], vertical_alignment="top")
    with colA:
        st.markdown("### Role-play scenario (finance-native)")
        st.markdown(
            """
You are **Mr. Alex Chen**, Risk Officer at **FinSecure Bank**.  
The bank uses AI models for (i) **trading signals** and (ii) **credit approvals**.  
Your job is to ensure these models remain within acceptable risk boundaries and are **audit-defensible** under internal governance (e.g., SR 11‑7 style monitoring) and external expectations (e.g., EU AI Act Article 12-style logging).
            """.strip()
        )

        callout(
            "Why audit logs exist (in one line)",
            "Because outcomes are not enough — you must be able to reconstruct *how* and *why* a decision happened, later, under scrutiny.",
            "info",
        )

        st.markdown("### What you’ll learn in this lab")
        st.markdown(
            """
- What must be captured in a decision audit log (and what question each field answers)  
- How monitoring checks become **alerts** written into the same log (evidence trail)  
- How a Risk Officer triages: “What changed?”, “How severe?”, “What requires review?”  
- How to produce a committee-ready report (evidence + sign-off)
            """.strip()
        )

        st.markdown("### The audit log: what you must capture (and why it matters)")
        st.markdown(
            """
**Inputs (reconstruction)** — What did the model *see*?  
**Outputs (impact)** — What decision did it *make*, and with what certainty?  
**Context (accountability)** — For whom / which portfolio / which model version?  
**Governance (control evidence)** — Was it flagged, reviewed, and documented?
            """.strip()
        )

        st.markdown("#### Field → investigative question mapping")
        st.markdown(
            """
- **timestamp** → *When did the decision occur? Was it during an incident window?*  
- **model_name + model_version** → *Which model produced it? Did version change?*  
- **input_hash + input_features** → *Can we reproduce the same input state?*  
- **prediction + decision_type** → *What action did the model recommend?*  
- **confidence** → *Did the model’s certainty shift materially?*  
- **explanation** → *What rationale was recorded for audit/committee review?*  
- **anomaly_flag + review_status + review_notes** → *Was it flagged? Was it reviewed? What was concluded?*
            """.strip()
        )

        st.markdown("### Micro-example (finance-native)")
        st.markdown(
            """
“On Feb 7, the trading model shifted from 10% SELL to 55% SELL.  
Without an audit log (inputs + model version + context), you cannot prove whether this was regime change, a silent model update, or a data/policy break.”
            """.strip()
        )

    with colB:
        st.markdown("### Controls")
        st.button("Reset demo (clear audit log)", on_click=reset_demo)

        st.markdown("### What good looks like (targets)")
        st.markdown(
            """
These are **learning targets** to anchor interpretation (not universal standards):

- **Unacknowledged HIGH alerts:** 0  
- **Pending reviews:** low and bounded by policy capacity  
- **Logging completeness (coverage):** high enough to reconstruct decisions
            """.strip()
        )

        decisions_df = get_decisions_df()
        cov = compute_logging_coverage(decisions_df)
        if cov.empty:
            st.caption("Logging coverage will appear after you generate decisions.")
        else:
            st.markdown("#### Logging completeness (coverage)")
            st.dataframe(
                cov.assign(Coverage=cov["Coverage"].map(pct)),
                use_container_width=True,
                hide_index=True,
            )
            low_cov = cov[cov["Coverage"] < 0.95]
            if not low_cov.empty:
                st.warning(
                    "Some key fields have <95% coverage. In a real governance setting, incomplete logging weakens audit defensibility."
                )

        with st.expander("How are decisions captured without changing the model? (concept-only)", expanded=False):
            st.markdown(
                """
In practice, firms often use a **logging wrapper** around model execution so that every decision emits:
- the input snapshot (or a hash + reference),
- the output + confidence,
- and governance metadata.

This keeps the **governance evidence** consistent even as models evolve.
                """.strip()
            )

# =============================================================================
# PAGE 2: SIMULATION & DATA GENERATION
# =============================================================================
elif st.session_state["page"] == "Simulation & Data Generation":
    st.header("Generate a Decision Stream (and Stress-Test Monitoring Rules)")

    callout(
        "What you’re doing here",
        "You will generate a production-like stream of decisions, then inject a controlled anomaly to test whether monitoring rules detect it.",
        "info",
    )

    st.markdown("### Simulation controls (chosen to make the anomaly visible)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.total_simulation_days = st.number_input("Horizon (days)", min_value=3, max_value=30, value=int(st.session_state.total_simulation_days))
    with c2:
        st.session_state.anomaly_trigger_day = st.number_input(
            "Inject anomaly on day # (1…Horizon)",
            min_value=1,
            max_value=int(st.session_state.total_simulation_days),
            value=int(st.session_state.anomaly_trigger_day),
            help="This is the day where we deliberately change behavior (e.g., SELL skew or looser credit approvals).",
        )
    with c3:
        st.session_state.n_trading_decisions_per_day = st.number_input("Trading signals per day (count)", min_value=10, max_value=500, value=int(st.session_state.n_trading_decisions_per_day))
    with c4:
        st.session_state.n_credit_decisions_per_day = st.number_input("Credit decisions per day (count)", min_value=10, max_value=1000, value=int(st.session_state.n_credit_decisions_per_day))

    assumptions_box([
        "Each row in the audit log is one decision event (one trade signal OR one credit application decision).",
        "The anomaly is intentional and designed to mimic real-world breakpoints (data feed issues, policy overrides, silent model changes).",
    ])

    st.markdown("### What anomaly gets injected?")
    st.markdown(
        """
- **Trading model:** a sudden distribution shift toward **SELL** decisions (behavioral regime shift)  
- **Credit model:** a shift toward more approvals by lowering the average FICO score (policy-like shift)
        """.strip()
    )

    checkpoint_question(
        "If you inject an anomaly on Day 4 of a 6‑day horizon, what is the governance purpose?",
        ["To maximize model accuracy", "To test whether monitoring detects a controlled behavioral break", "To train the model"],
        "To test whether monitoring detects a controlled behavioral break",
        "In governance, simulations with injected anomalies are used to validate *controls* (detection + workflow), not to improve predictive power.",
    )

    st.divider()
    if st.button("Generate decisions and write to audit log", type="primary"):
        try:
            total_days = int(st.session_state.total_simulation_days)
            anomaly_day = int(st.session_state.anomaly_trigger_day)
            for day in range(1, total_days + 1):
                simulate_production_day(
                    logger_instance=st.session_state.logger,
                    n_trading=int(st.session_state.n_trading_decisions_per_day),
                    n_credit=int(st.session_state.n_credit_decisions_per_day),
                    anomaly_day=(day == anomaly_day),
                )
            invalidate_caches()
            st.success("Decisions generated and logged.")
        except Exception as e:
            st.error(f"Simulation failed: {e}")

    decisions_df = get_decisions_df()
    st.markdown("### Simulated Decisions Overview (audit log preview)")
    st.caption("Each row = one logged decision event. Your goal is to ensure decisions are reconstructable and reviewable.")
    if decisions_df.empty:
        st.info("No decisions yet. Click **Generate decisions and write to audit log**.")
    else:
        st.write(f"Total logged decisions: **{len(decisions_df):,}**")
        st.dataframe(decisions_df.sort_values("timestamp", ascending=False).head(25), use_container_width=True)

        st.markdown("#### Logging completeness (coverage)")
        cov = compute_logging_coverage(decisions_df)
        st.dataframe(cov.assign(Coverage=cov["Coverage"].map(pct)), use_container_width=True, hide_index=True)

        evidence_box([
            "Coverage % for required fields (inputs, outputs, model version, user/portfolio context)",
            "Any missing-field exceptions and their resolution",
            "Retention policy and access control statement (conceptually)",
        ])

# =============================================================================
# PAGE 3: ANOMALY DETECTION
# =============================================================================
elif st.session_state["page"] == "Anomaly Detection":
    st.header("Monitoring Rules (Transparent, Auditable Checks)")

    callout(
        "Key idea",
        "These checks are *smoke detectors*. A triggered alert is a triage signal — not proof of model failure.",
        "info",
    )

    st.markdown("### Choose monitoring windows")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.recent_window_days = st.slider("Recent window (days)", min_value=1, max_value=7, value=int(st.session_state.recent_window_days))
    with c2:
        st.session_state.baseline_window_days = st.slider("Baseline window (days)", min_value=2, max_value=30, value=int(st.session_state.baseline_window_days))

    assumptions_box([
        "Baseline window is intended to represent 'normal' behavior.",
        "Recent window is the period you want to test for sudden behavior change.",
        "Thresholds below are demo values; in practice they reflect risk appetite + expected variability.",
    ])

    st.markdown("### Checks and formulas (do not treat as black boxes)")
    # IMPORTANT: Keep all formulae markdown lines from the original app (do not remove).
    st.markdown("#### 1) Distribution shift (Trading + Credit)")
    st.markdown(r"""
$$
\Delta P_d = |P_{\text{recent}}(d) - P_{\text{baseline}}(d)|
$$""")
    st.markdown(f"An alert is triggered if $\\Delta P_d > 0.20$ (20 percentage points) for any decision category $d$.")

    st.markdown("#### 2) Concentration risk (Trading only)")
    st.markdown(r"""
$$
\text{Max Concentration} = \max_{\text{entity}} \left( \frac{\text{Count of decisions for entity}}{\text{Total decisions in recent window}} \right)
$$""")
    st.markdown("An alert is triggered if Max Concentration > 0.30 (30%).")

    st.markdown("#### 3) Confidence shift (Trading + Credit)")
    st.markdown(r"""
$$
\Delta \text{Confidence} = |\text{mean Confidence}_{\text{recent}} - \text{mean Confidence}_{\text{baseline}}|
$$""")
    st.markdown(r"An alert is triggered if $\Delta \text{Confidence} > 0.10$.")

    st.markdown("#### 4) Volume anomaly (Trading + Credit)")
    st.markdown(r"""
$$
\text{Daily Count}_{\text{recent}} > 2 \times \text{Daily Count}_{\text{baseline}}
$$""")
    st.markdown(r"""
$$
\text{Daily Count}_{\text{recent}} < 0.3 \times \text{Daily Count}_{\text{baseline}}
$$""")
    st.markdown("An alert is triggered if either condition is met.")

    st.divider()

    st.markdown("### Risk appetite intuition (how to think about thresholds)")
    st.markdown(
        """
- **Tighter thresholds** → faster detection, more false alarms (more investigation workload)  
- **Looser thresholds** → fewer false alarms, slower detection (risk can accumulate unnoticed)
        """.strip()
    )

    checkpoint_question(
        "If SELL share rises from 15% to 40%, does it breach the distribution shift threshold (20pp)?",
        ["Yes", "No", "Not enough information"],
        "Yes",
        "ΔP = |0.40 − 0.15| = 0.25 (25 percentage points) which is > 0.20, so it triggers.",
    )

    st.divider()

    decisions_df = get_decisions_df()
    models = sorted(decisions_df["model_name"].dropna().unique().tolist()) if not decisions_df.empty and "model_name" in decisions_df.columns else []

    if st.button("Run checks and write alerts to the audit log", type="primary"):
        try:
            if not models:
                st.warning("No model names found yet. Generate decisions first.")
            else:
                for m in models:
                    detect_decision_anomalies(
                        logger_instance=st.session_state.logger,
                        model_name=m,
                        window_days=int(st.session_state.recent_window_days),
                        baseline_days=int(st.session_state.baseline_window_days),
                    )
                invalidate_caches()
                st.success("Anomaly checks completed and alerts (if any) were written to the log.")
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")

    alerts_df = get_alerts_df()
    st.markdown("### Alerts written to the audit log (what needs triage)")
    st.caption("An alert is a triage ticket: investigate, document, and close the loop.")
    if alerts_df.empty:
        st.info("No alerts recorded yet. Generate decisions, then run anomaly checks.")
    else:
        alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"], errors="coerce")
        st.dataframe(alerts_df.sort_values("timestamp", ascending=False), use_container_width=True)

        unack_df = alerts_df[alerts_df.get("acknowledged", 0) == 0]
        high_unack = unack_df[unack_df.get("severity", "").astype(str).str.upper() == "HIGH"]
        if not high_unack.empty:
            st.warning("There are **unacknowledged HIGH** alerts. In governance terms, detection exists but response ownership is missing.")

        st.markdown("#### Triage ownership (acknowledge an alert)")
        a1, a2, a3 = st.columns([1, 2, 2])
        with a1:
            alert_id = st.number_input("Alert ID", min_value=1, value=int(alerts_df["id"].max()))
        with a2:
            acknowledged_by = st.text_input("Acknowledged by (name/role)", value="Alex Chen (Risk Officer)")
        with a3:
            if st.button("Acknowledge selected alert"):
                acknowledge_alert(int(alert_id), acknowledged_by)

        evidence_box([
            "Alert SLA policy (how quickly HIGH/MEDIUM must be triaged)",
            "Standard triage notes template: hypothesis → evidence checked → conclusion → actions",
            "Link from alerts to impacted decisions and versions",
        ])

# =============================================================================
# PAGE 4: RISK OFFICER DASHBOARD
# =============================================================================
elif st.session_state["page"] == "Risk Officer Dashboard":
    st.header("Risk Officer Dashboard: Daily Triage Snapshot → Investigation → Documentation")

    decisions_df = get_decisions_df()
    alerts_df = get_alerts_df()

    if decisions_df.empty:
        st.info("No decisions available yet. Generate decisions in **Simulation & Data Generation**.")
        st.stop()

    decisions_df["timestamp"] = pd.to_datetime(decisions_df["timestamp"], errors="coerce")

    # Sidebar controls
    st.sidebar.divider()
    st.sidebar.subheader("Dashboard Controls")
    model_names = sorted(decisions_df["model_name"].dropna().unique().tolist()) if "model_name" in decisions_df.columns else []
    if not model_names:
        model_names = ["MomentumSignal", "CreditScorer"]

    # If not set, default to first model
    if st.session_state.selected_dashboard_model is None:
        st.session_state.selected_dashboard_model = model_names[0]

    st.session_state.selected_dashboard_model = st.sidebar.selectbox(
        "Model",
        model_names,
        index=model_names.index(st.session_state.selected_dashboard_model) if st.session_state.selected_dashboard_model in model_names else 0,
    )
    st.session_state.dashboard_review_period = st.sidebar.slider("Lookback window (days)", min_value=3, max_value=30, value=int(st.session_state.dashboard_review_period))

    model = st.session_state.selected_dashboard_model
    lookback_days = int(st.session_state.dashboard_review_period)

    end_ts = decisions_df["timestamp"].max()
    start_ts = end_ts - timedelta(days=lookback_days)
    ddf = decisions_df[(decisions_df["timestamp"] >= start_ts) & (decisions_df["model_name"] == model)].copy()

    if ddf.empty:
        st.warning("No decisions in the selected lookback window for this model.")
        st.stop()

    # Summary stats
    st.markdown("### Daily Triage Snapshot")
    total = len(ddf)
    pending = (ddf.get("review_status", "").astype(str).str.lower() == "pending").sum() if "review_status" in ddf.columns else 0
    flagged = (ddf.get("anomaly_flag", 0) == 1).sum() if "anomaly_flag" in ddf.columns else 0

    # Alerts slice for this model + window
    if not alerts_df.empty and "timestamp" in alerts_df.columns:
        alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"], errors="coerce")
        adf = alerts_df[(alerts_df["timestamp"] >= start_ts) & (alerts_df.get("model_name", "") == model)].copy()
    else:
        adf = pd.DataFrame()

    unack = (adf.get("acknowledged", 0) == 0).sum() if not adf.empty else 0

    days_span = max(1, ddf["timestamp"].dt.normalize().nunique())
    avg_per_day = total / days_span

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decisions logged (period)", f"{total:,}")
    m2.metric("Throughput (avg/day)", f"{avg_per_day:,.1f}")
    m3.metric("Decisions awaiting human sign-off", f"{pending:,}")
    m4.metric("Alerts not yet triaged", f"{unack:,}")

    # Governance guardrails
    if unack > 0:
        st.warning("Guardrail: unacknowledged alerts indicate a governance backlog. Detection without response is an ineffective control.")
    if pending > 0 and pending / total > 0.25:
        st.warning("Guardrail: a high pending-review ratio suggests monitoring capacity is insufficient relative to throughput.")

    st.divider()

    # Chart: Decision distribution over time
    st.markdown("### Did the model change its behavior?")
    if "prediction" in ddf.columns:
        daily = (
            ddf.assign(day=ddf["timestamp"].dt.date)
               .groupby(["day", "prediction"])
               .size()
               .reset_index(name="count")
        )
        daily_tot = daily.groupby("day")["count"].transform("sum")
        daily["pct"] = daily["count"] / daily_tot

        fig = px.bar(daily, x="day", y="pct", color="prediction", title="Daily Decision Distribution (%)")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Decision translation: if a category (e.g., SELL or APPROVE) rises sharply, investigate whether this reflects a market regime shift, a model/version change, or a data/policy break.")
    else:
        st.info("No prediction field available to plot distribution.")

    # Confidence distribution
    st.markdown("### Did certainty change (possible drift or regime shift)?")
    if "confidence" in ddf.columns:
        fig2 = px.histogram(ddf, x="confidence", nbins=20, title="Distribution of Prediction Confidence Scores")
        st.plotly_chart(fig2, use_container_width=True)

        recent_cut = end_ts - timedelta(days=min(2, lookback_days))
        recent_mean = pd.to_numeric(ddf[ddf["timestamp"] >= recent_cut]["confidence"], errors="coerce").dropna().mean()
        base_mean = pd.to_numeric(ddf[ddf["timestamp"] < recent_cut]["confidence"], errors="coerce").dropna().mean()
        st.caption(f"Recent mean confidence: **{recent_mean:.3f}** • Baseline mean confidence: **{base_mean:.3f}**")
        st.caption("Decision translation: if confidence collapses, tighten risk buffers (smaller positions / higher review sampling / tighter credit cutoffs) until you confirm cause.")
    else:
        st.info("No confidence field available to plot.")

    # Concentration risk (ticker-based; trading-like)
    if "ticker" in ddf.columns and ddf["ticker"].notna().any():
        st.markdown("### Is the model over-focusing on a few names?")
        recent_window = ddf[ddf["timestamp"] >= (end_ts - timedelta(days=min(2, lookback_days)))].copy()
        if not recent_window.empty:
            top = recent_window["ticker"].value_counts(normalize=True).head(10).reset_index()
            top.columns = ["ticker", "share"]
            fig3 = px.bar(top, x="ticker", y="share", title="Top Tickers by Share of Decisions (Recent Window)")
            fig3.add_hline(y=0.30, line_dash="dash")
            fig3.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Decision translation: if concentration rises, apply portfolio constraints or require human approval for repeated same-name actions.")
            st.caption("Watch-out: decision concentration is not the same as exposure concentration—confirm with portfolio weights.")
        else:
            st.info("Not enough recent data to compute concentration.")
    else:
        st.caption("Concentration risk chart applies to trading-like models (requires an entity field such as ticker).")

    # Alert timeline
    st.markdown("### When did monitoring rules fire, and how severe?")
    if not adf.empty:
        fig4 = px.scatter(adf, x="timestamp", y="severity", color="severity", title="Alert Timeline by Severity")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Decision translation: clusters of HIGH alerts warrant escalation (incident response mindset): verify data feeds, freeze changes, increase manual sampling.")
    else:
        st.info("No alerts in the selected window for this model.")

    st.divider()

    # Flagged decisions
    st.markdown("### Decisions marked for review (anomaly flag) — human sign-off workflow")
    st.caption("Workflow: open a flagged decision → verify inputs → compare to baseline → document conclusion → mark reviewed.")

    if "anomaly_flag" in ddf.columns and (ddf["anomaly_flag"] == 1).any():
        flagged_df = ddf[ddf["anomaly_flag"] == 1].sort_values("timestamp", ascending=False).copy()
        st.dataframe(flagged_df.head(50), use_container_width=True)

        with st.expander("Common misconceptions & watch-outs", expanded=False):
            st.markdown(
                """
- A flagged decision is **not** automatically incorrect; it is a **triage signal**.
- A legitimate regime change can trigger alerts; governance requires documentation, not denial.
- A “reviewed” label without rationale is not defensible under audit.
                """.strip()
            )
    else:
        st.info("No decisions are flagged for review in this window.")

# =============================================================================
# PAGE 5: AUDIT REPORT
# =============================================================================
elif st.session_state["page"] == "Audit Report":
    st.header("Weekly/Monthly AI Governance Report (Evidence + Sign-Off)")

    decisions_df = get_decisions_df()
    alerts_df = get_alerts_df()

    st.sidebar.divider()
    st.sidebar.subheader("Report Controls")
    st.session_state.audit_report_period = st.sidebar.slider("Reporting period (days)", min_value=7, max_value=90, value=int(st.session_state.audit_report_period))
    period_days = int(st.session_state.audit_report_period)

    if decisions_df.empty:
        st.info("No decisions available yet. Generate decisions first.")
        st.stop()

    st.markdown("### What this report is (and is not)")
    st.markdown(
        """
- **Is:** a governance artifact for committee/regulators: evidence of monitoring, triage, and sign-off  
- **Is not:** a performance report or a claim that the model is “correct”
        """.strip()
    )

    callout(
        "Guardrail",
        "A report that shows HIGH alerts with no documented triage or review notes is a governance failure — even if P&L looked fine.",
        "warning",
    )

    # Generate report text using source.py function (no implementation detail exposed)
    if st.button("Generate report (committee-ready)", type="primary"):
        try:
            report_dict = generate_audit_report(logger_instance=st.session_state.logger, period_days=period_days)
            st.session_state["last_report_dict"] = report_dict
            st.success("Report generated.")
        except Exception as e:
            st.error(f"Report generation failed: {e}")

    report_dict = st.session_state.get("last_report_dict", None)

    # Executive snapshot for interpretation (simple, evidence-based)
    decisions_df["timestamp"] = pd.to_datetime(decisions_df["timestamp"], errors="coerce")
    end_ts = decisions_df["timestamp"].max()
    start_ts = end_ts - timedelta(days=period_days)
    period_df = decisions_df[decisions_df["timestamp"] >= start_ts].copy()

    pending = (period_df.get("review_status", "").astype(str).str.lower() == "pending").sum() if "review_status" in period_df.columns else 0
    flagged = (period_df.get("anomaly_flag", 0) == 1).sum() if "anomaly_flag" in period_df.columns else 0
    total = len(period_df)

    if not alerts_df.empty and "timestamp" in alerts_df.columns:
        alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"], errors="coerce")
        period_alerts = alerts_df[alerts_df["timestamp"] >= start_ts].copy()
        unack = (period_alerts.get("acknowledged", 0) == 0).sum()
    else:
        period_alerts = pd.DataFrame()
        unack = 0

    st.markdown("### Executive summary (period)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Decisions logged", f"{total:,}")
    a2.metric("Flagged for review", f"{flagged:,}")
    a3.metric("Pending reviews", f"{pending:,}")
    a4.metric("Unacknowledged alerts", f"{unack:,}")

    st.markdown("### Evidence: logging completeness (coverage)")
    cov = compute_logging_coverage(period_df)
    if cov.empty:
        st.info("Coverage will appear after decisions are generated.")
    else:
        st.dataframe(cov.assign(Coverage=cov["Coverage"].map(pct)), use_container_width=True, hide_index=True)

    evidence_box([
        "Coverage % for required fields (inputs, outputs, model version, user/portfolio context)",
        "Alert triage SLA compliance (especially HIGH)",
        "Review note completeness for flagged decisions",
        "Retention and access controls statement (conceptual controls)",
    ])

    st.divider()
    if report_dict:
        # Convert dict to formatted text
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"       {report_dict['report_title']}")
        report_lines.append(f"       Period: {report_dict['period']}")
        report_lines.append(f"       Generated: {report_dict['generation_date']}")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("--- EXECUTIVE SUMMARY ---")
        summary = report_dict['executive_summary']
        report_lines.append(f"Total AI Decisions: {summary['total_ai_decisions']}")
        report_lines.append(f"Active Models: {summary['models_active']}")
        report_lines.append(f"Total Alerts: {summary['total_alerts']} (HIGH: {summary['high_severity_alerts']}, Unacknowledged: {summary['unacknowledged_alerts']})")
        report_lines.append(f"Decisions Pending Review: {summary['decisions_pending_review']}")
        report_lines.append("")
        report_lines.append("--- MODEL SUMMARIES ---")
        if not report_dict['model_summaries']:
            report_lines.append("No model summaries for the period.")
        for model, stats in report_dict['model_summaries'].items():
            report_lines.append(f"\nModel: {model}")
            report_lines.append(f"  Decisions: {stats['decision_count']}")
            report_lines.append(f"  Distribution: {stats['decision_distribution']}")
            report_lines.append(f"  Avg Confidence: {stats['avg_confidence']}")
            report_lines.append(f"  Alerts: {stats['alerts_count']}, Flagged Decisions: {stats['anomaly_flags_count']}")
        report_lines.append("")
        report_lines.append("--- REGULATORY COMPLIANCE STATUS ---")
        for reg, status in report_dict['regulatory_compliance'].items():
            report_lines.append(f"  {reg}: {status}")
        report_lines.append("")
        report_lines.append("--- SIGN-OFF ---")
        for role, details in report_dict['sign_off'].items():
            report_lines.append(f"{role}: {details['name']} (Date: {details['date']})")
            report_lines.append("  Signature: __________________________\n")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        st.text_area("Audit report (preview)", value=report_text, height=460)
        buf = io.BytesIO(report_text.encode("utf-8"))
        st.download_button(
            label="Download report (txt)",
            data=buf,
            file_name=f"audit_report_{start_ts.date()}_{end_ts.date()}.txt",
            mime="text/plain",
        )
    else:
        st.caption("Click **Generate report (committee-ready)** to produce the governance artifact.")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
