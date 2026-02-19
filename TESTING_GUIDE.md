# Testing Guide - Audit Logging Demo

## Quick Test Scenario

### Step 1: Verify Multi-User Sessions
1. Open the Streamlit app in your browser
2. Look at the **sidebar** - you should see:
   ```
   🔒 Session DB: finsecure_audit_a1b2c3d4.db
   Each user has an isolated database session
   ```
3. Open the app in a **new browser tab** (or incognito window)
4. Notice the session DB hash is **different** - this means isolated sessions! ✅

### Step 2: Generate Decisions and Trigger Alerts

1. **Navigate to:** "Simulation & Data Generation"
2. **Set parameters:**
   - Horizon: 6 days
   - Inject anomaly on day: 4
   - Trading signals per day: 50
   - Credit decisions per day: 100

3. **Click:** "Generate decisions and write to audit log"
   - Should see: ✅ "Decisions generated and logged."
   - Should see a table with ~900 decisions (6 × 150)

### Step 3: Run Anomaly Detection

1. **Navigate to:** "Anomaly Detection"
2. **Set monitoring windows:**
   - Recent window: 2 days
   - Baseline window: 4 days

3. **Click:** "Run checks and write alerts to the audit log"

4. **Expected Results - You should now see alerts like:**

   | Severity | Type | Description |
   |----------|------|-------------|
   | HIGH | distribution_shift | 'MomentumSignal': 'SELL' prediction shifted from 33% to 70% (delta=+37%) |
   | MEDIUM | confidence_shift | 'MomentumSignal': avg confidence shifted from 0.72 to 0.65 |
   | MEDIUM | distribution_shift | 'CreditScorer': 'APPROVE' prediction shifted from 52% to 67% (delta=+15%) |

### Step 4: Triage Dashboard

1. **Navigate to:** "Risk Officer Dashboard"
2. **Select model:** MomentumSignal (dropdown)
3. **Review charts:**
   - Daily Decision Distribution - should show spike in SELL on anomaly day
   - Confidence histogram - should show shift
   - Alert timeline - should show multiple alerts on recent days

4. **Acknowledge an alert:**
   - Find an alert ID from the table
   - Enter your name: "Alex Chen (Risk Officer)"
   - Click "Acknowledge selected alert"
   - ✅ Should see confirmation and alert marked as acknowledged

### Step 5: Generate Audit Report

1. **Navigate to:** "Audit Report"
2. **Set period:** 14 days
3. **Click:** "Generate report (committee-ready)"
4. **Review report sections:**
   - Executive summary (should show unacknowledged alerts count)
   - Model summaries (decision counts, distribution)
   - Regulatory compliance status
   - Sign-off section

5. **Download report:** Click "Download report (txt)"

## What's Fixed

### ✅ Database Readonly Error - FIXED
- **Before:** "attempt to write a readonly database"
- **After:** Each user has isolated database with proper WAL mode

### ✅ No Alerts Generated - FIXED
- **Before:** Thresholds too high, no alerts triggered
- **After:** Lowered thresholds (15pp for distribution shift), multiple alerts generated

### ✅ Multi-User Conflicts - FIXED
- **Before:** All users shared one database (conflicts)
- **After:** Each session gets unique database file

## Troubleshooting

### If you still don't see alerts:
1. Make sure anomaly day (4) is between baseline and recent windows
2. Try lowering recent window to 1 day
3. Check terminal output for detection logs

### If you get database errors:
1. Click "Reset demo" button on Home page
2. Refresh browser (Ctrl+R)
3. Check file permissions in terminal:
   ```bash
   ls -la finsecure_audit_*.db
   ```

### To clean up old session databases:
```bash
cd /home/user1/QuCreate/QuLabs/69935a632f07e05003271940
rm -f finsecure_audit_*.db*
```

## Session Cleanup (Optional)

To enable automatic database deletion when users close their browser:

1. Open `app.py`
2. Find the `cleanup_session()` function (around line 115)
3. Uncomment these lines:
   ```python
   # import os
   # if os.path.exists(st.session_state.logger.db_path):
   #     os.remove(st.session_state.logger.db_path)
   ```

**Note:** This will delete audit logs when sessions end. Use only for demo purposes!
