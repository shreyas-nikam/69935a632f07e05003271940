# Fix: Decisions Not Being Generated

## Problem
After implementing session-specific databases, decisions were not being written to the correct database file because:

1. The decorated model functions (`generate_trading_signal`, `score_credit_application`) were using a **hardcoded global logger** pointing to `finsecure_ai_audit_log.db`
2. Each session created a unique database like `finsecure_audit_{hash}.db` 
3. The mismatch meant decisions were written to the wrong database

## Solution
Added a `set_global_logger()` function to dynamically update the global logger reference:

### Changes Made:

1. **source.py** - Added setter function:
```python
def set_global_logger(logger_instance: AIDecisionLogger):
    """Update the global logger to use a session-specific logger instance."""
    global _global_decision_logger
    _global_decision_logger = logger_instance
```

2. **source.py** - Update global logger in `simulate_production_day()`:
```python
def simulate_production_day(logger_instance: AIDecisionLogger, ...):
    # Update global logger to use the session-specific logger
    set_global_logger(logger_instance)
    # ... rest of function
```

3. **app.py** - Set global logger on session initialization:
```python
def initialize_session_state():
    if "logger" not in st.session_state:
        db_path = get_user_db_path()
        st.session_state["logger"] = AIDecisionLogger(db_path=db_path)
        st.session_state.logger._create_tables()
        # Set the global logger to use this session's logger
        set_global_logger(st.session_state.logger)
```

4. **app.py** - Also update on reset:
```python
def reset_demo():
    # ... create new logger ...
    set_global_logger(st.session_state.logger)
```

5. **app.py** - Added progress indicator and better error reporting:
```python
with st.spinner(f"Generating decisions for {total_days} days..."):
    # simulation code
```

## Result
✅ Decisions are now correctly written to the session-specific database  
✅ Multiple users can generate decisions without conflicts  
✅ Better error messages with full traceback if something fails  
✅ Visual feedback with spinner during generation

## Testing
1. Refresh the Streamlit app
2. Click "Generate decisions and write to audit log"
3. You should see:
   - A spinner showing progress
   - Success message: "Decisions generated and logged."
   - Table showing ~900 decisions (6 days × 150 decisions/day)

## Technical Note
The decorator pattern requires a global logger reference at module load time. Rather than refactoring the entire decorator system, we dynamically update the global reference to point to the session-specific logger before each simulation run.
