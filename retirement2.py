import json
import os
import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retirement Income Bridge (Couple)", layout="wide")

# ======================
# Settings persistence
# ======================
SETTINGS_FILE = "retirement_settings.json"

# ----------------------
# 2026 defaults (Ontario + Federal) — as you had
# NOTE: This model is a simplified planner-style tax estimate (not a CRA calculator).
# ----------------------
# Defaults imported from simulation module to ensure consistency
from retirement2_helpers import DEFAULTS
from retirement_simulation import  run_model_simulation

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            merged = DEFAULTS.copy()
            if isinstance(loaded, dict):
                # Migration: If current_year_goal_income missing (old file), sync with goal_income
                if "current_year_goal_income" not in loaded and "goal_income" in loaded:
                    loaded["current_year_goal_income"] = loaded["goal_income"]
                merged.update(loaded)
            return merged
        except Exception:
            return DEFAULTS.copy()
    return DEFAULTS.copy()

def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as ex:
        st.sidebar.warning(f"Could not save settings: {ex}")

def collect_settings_from_session() -> dict:
    return {k: st.session_state.get(k, DEFAULTS[k]) for k in DEFAULTS.keys()}

def on_any_change():
    save_settings(collect_settings_from_session())

# ----------------------
# Initialize session_state ONCE
# ----------------------
if "___initialized" not in st.session_state:
    settings = load_settings()
    for k, v in settings.items():
        st.session_state[k] = v
    st.session_state["___initialized"] = True

# ----------------------
# Header
# ----------------------
st.title("Retirement Income Bridge — Couple (RRIF/TFSA/NonReg + Taxes)")
st.caption(
    "All amounts are in today's dollars. Uses a REAL return (beyond inflation). "
    "Planner-style tax estimate (Ontario+Federal, credits, surtax, ON health premium). "
    "Bucket strategy: fill RRIF to a taxable-income target, then TFSA, then NonReg; "
    "surplus after-tax cash is transferred to TFSA (room permitting) then NonReg."
)

# ======================
# Inputs
# ======================
with st.sidebar:
    st.header("Settings Import/Export")
    
    # --- DOWNLOAD ---
    settings_data = collect_settings_from_session()
    json_str = json.dumps(settings_data, indent=2)
    st.download_button(
        label="Download Current Settings (JSON)",
        data=json_str,
        file_name="retirement_settings.json",
        mime="application/json",
    )
    
    # --- UPLOAD ---
    uploaded_file = st.file_uploader("Upload Settings (JSON)", type=["json"])
    if uploaded_file is not None:
        try:
            new_settings = json.load(uploaded_file)
            if isinstance(new_settings, dict):
                # Update session state with new values
                for k, v in new_settings.items():
                    if k in DEFAULTS or k.startswith("___"):
                        st.session_state[k] = v
                
                # Persistence + Refresh
                save_settings(collect_settings_from_session())
                st.success("Settings uploaded! Refreshing...")
                st.rerun()
            else:
                st.error("Invalid settings file format.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.divider()
    st.header("Timeline")
    st.number_input("Start year", min_value=1900, max_value=2100, key="start_year", on_change=on_any_change)
    st.number_input("Your age in start year", min_value=0, max_value=120, key="your_start_age", on_change=on_any_change)
    st.number_input("Spouse age difference (spouse - you)", min_value=-20, max_value=20, key="spouse_age_diff", on_change=on_any_change)
    st.slider("Years to model", min_value=5, max_value=60, key="years", on_change=on_any_change)

    # ======================
    # Mortality assumptions (NEW)
    # ======================
    st.subheader("Mortality assumptions")
    st.number_input("Your assumed death age", min_value=0, max_value=120, key="your_death_age", on_change=on_any_change)
    st.number_input("Spouse assumed death age", min_value=0, max_value=120, key="spouse_death_age", on_change=on_any_change)

    st.header("Buckets (Household Totals)")
    st.number_input("RRIF Start Balance", min_value=0, step=1000, key="rrif_start_balance", on_change=on_any_change)
    st.number_input("TFSA Start Balance", min_value=0, step=1000, key="tfsa_start_balance", on_change=on_any_change)
    st.number_input("Non-Reg Start Balance", min_value=0, step=1000, key="nonreg_start_balance", on_change=on_any_change)

    st.subheader("TFSA Room Settings")
    st.caption("Start room is household TOTAL. Annual new room is PER PERSON (×2).")
    st.number_input("Total TFSA room available NOW (household)", min_value=0, step=1000, key="tfsa_room_start_total", on_change=on_any_change)
    st.number_input("TFSA annual new room (per person)", min_value=0, step=100, key="tfsa_room_annual_per_person", on_change=on_any_change)

    st.header("RRIF Strategy")
    st.selectbox("RRIF Withdrawal Strategy", 
                 ["Dynamic Optimization", "Fixed Taxable Income Target"], 
                 index=0 if st.session_state.get("rrif_strategy_mode", "dynamic") == "dynamic" else 1,
                 key="_rrif_strat_ui",
                 on_change=lambda: st.session_state.update({"rrif_strategy_mode": "dynamic" if st.session_state._rrif_strat_ui == "Dynamic Optimization" else "custom_target", "___initialized": True}) or on_any_change())
    
    is_dynamic = (st.session_state.get("rrif_strategy_mode", "dynamic") == "dynamic")
    
    if is_dynamic:
        st.caption("Optimizes RRIF withdrawals based on projected future tax pressure.")
        st.number_input("Peak rate safety margin (decimal, e.g. 0.08)", step=0.01, format="%.3f", key="dynamic_melt_safety_margin", on_change=on_any_change)
        with st.expander("Advanced Strategy Settings"):
            st.number_input("Max marginal rate guardrail (decimal, e.g. 0.54)", step=0.01, format="%.3f", key="dynamic_max_marginal_rate", on_change=on_any_change)
    else:
        st.caption("Fills RRIF withdrawal to a fixed taxable income target each year.")
        st.number_input("RRIF taxable target per person", min_value=0, step=1000, key="rrif_taxable_target_per_person", on_change=on_any_change)

    # st.number_input("RRIF taxable target per person (bracket-fill)", min_value=0, step=1000, key="rrif_taxable_target_per_person", on_change=on_any_change)
    st.checkbox("Allow surplus transfers (RRIF → TFSA/NonReg)", key="allow_surplus_transfers", on_change=on_any_change)

    st.header("Return + Goal")
    st.number_input("Real return (beyond inflation) %", step=0.25, key="real_return_pct", on_change=on_any_change)
    st.number_input("Household goal income - Current Year (Year 1)", min_value=0, step=1000, key="current_year_goal_income", on_change=on_any_change)
    st.number_input("Household goal income - Future Years", min_value=0, step=1000, key="goal_income", on_change=on_any_change)
    st.checkbox("Treat goal income as AFTER-TAX (net spend)", key="goal_is_after_tax", on_change=on_any_change)

    st.header("Your incomes (annual, today's $)")
    st.number_input("Your CPP starts at age", min_value=0, max_value=120, key="your_cpp_start_age", on_change=on_any_change)
    st.number_input("Your CPP amount", min_value=0, step=500, key="your_cpp_amount", on_change=on_any_change)
    st.number_input("Your OAS starts at age", min_value=0, max_value=120, key="your_oas_start_age", on_change=on_any_change)
    st.number_input("Your OAS amount", min_value=0, step=500, key="your_oas_amount", on_change=on_any_change)
    st.number_input("Your pension starts at age", min_value=0, max_value=120, key="your_pension_start_age", on_change=on_any_change)
    st.number_input("Your pension amount (starting value)", min_value=0, step=500, key="your_pension_amount", on_change=on_any_change)
    st.number_input("Your pension decline per year %", step=0.25, key="your_pension_decline_pct", on_change=on_any_change)

    st.header("Spouse incomes (annual, today's $)")
    st.checkbox("Manually set spouse age in start year", key="spouse_start_age_override", on_change=on_any_change)
    if st.session_state["spouse_start_age_override"]:
        st.number_input("Spouse age in start year", min_value=0, max_value=120, key="spouse_start_age", on_change=on_any_change)
    else:
        spouse_start_age_auto = int(st.session_state["your_start_age"]) + int(st.session_state["spouse_age_diff"])
        st.caption(f"Spouse age in start year (auto): {spouse_start_age_auto}")

    st.number_input("Spouse CPP starts at age", min_value=0, max_value=120, key="spouse_cpp_start_age", on_change=on_any_change)
    st.number_input("Spouse CPP amount", min_value=0, step=500, key="spouse_cpp_amount", on_change=on_any_change)
    st.number_input("Spouse OAS starts at age", min_value=0, max_value=120, key="spouse_oas_start_age", on_change=on_any_change)
    st.number_input("Spouse OAS amount", min_value=0, step=500, key="spouse_oas_amount", on_change=on_any_change)
    st.number_input("Spouse pension starts at age", min_value=0, max_value=120, key="spouse_pension_start_age", on_change=on_any_change)
    st.number_input("Spouse pension amount (starting value)", min_value=0, step=500, key="spouse_pension_amount", on_change=on_any_change)
    st.number_input("Spouse pension decline per year %", step=0.25, key="spouse_pension_decline_pct", on_change=on_any_change)

    st.header("Withdrawal rules")
    st.checkbox("Withdraw at start of year", key="withdraw_at_start_of_year", on_change=on_any_change)

    st.header("Taxes (Ontario + Federal)")
    st.checkbox("Enable tax model", key="enable_tax", on_change=on_any_change)
    st.checkbox("Enable OAS Clawback Model (15% add-on)", key="enable_oas_clawback", on_change=on_any_change)
    if st.session_state.get("enable_oas_clawback", True):
        st.number_input("OAS Clawback Threshold", min_value=0, step=1000, key="oas_clawback_threshold", on_change=on_any_change)
    st.slider("Split of spending withdrawals allocated to YOU (%)", 0.0, 100.0, key="withdraw_split_to_you_pct", on_change=on_any_change)
    st.slider("NonReg withdrawal taxable % (cap gains approx)", 0.0, 100.0, key="nonreg_withdraw_taxable_pct", on_change=on_any_change)
    st.checkbox("Treat RRIF as eligible pension income at 65+ (simplified)", key="treat_rrif_as_eligible_pension_income_at_65", on_change=on_any_change)

    with st.expander("Tax bracket tables (advanced)"):
        st.caption("Edit as JSON list: [[top_income, rate], ...]. Use a huge number (e.g., 1e18) for last top.")
        st.text_area("Federal brackets JSON", key="federal_brackets_json", height=170, on_change=on_any_change)
        st.text_area("Ontario brackets JSON", key="ontario_brackets_json", height=170, on_change=on_any_change)
        st.checkbox("Ontario surtax enabled", key="ont_surtax_enabled", on_change=on_any_change)
        st.number_input("Ontario surtax threshold 1 (basic ON tax)", step=1.0, key="ont_surtax_t1", on_change=on_any_change)
        st.number_input("Ontario surtax threshold 2 (basic ON tax)", step=1.0, key="ont_surtax_t2", on_change=on_any_change)
        st.checkbox("Ontario Health Premium enabled", key="ont_health_premium_enabled", on_change=on_any_change)

    with st.expander("Tax credits (advanced)"):
        st.number_input("Federal basic personal amount", step=10.0, key="fed_basic_personal_amount", on_change=on_any_change)
        st.number_input("Ontario basic personal amount", step=10.0, key="ont_basic_personal_amount", on_change=on_any_change)
        st.number_input("Federal age amount max (65+)", step=10.0, key="fed_age_amount_max", on_change=on_any_change)
        st.number_input("Federal age amount threshold", step=10.0, key="fed_age_amount_thresh", on_change=on_any_change)
        st.number_input("Federal age amount cutoff", step=10.0, key="fed_age_amount_cutoff", on_change=on_any_change)
        st.number_input("Ontario age amount max (65+)", step=10.0, key="ont_age_amount_max", on_change=on_any_change)
        st.number_input("Ontario age amount threshold", step=10.0, key="ont_age_amount_thresh", on_change=on_any_change)
        st.number_input("Ontario age amount cutoff", step=10.0, key="ont_age_amount_cutoff", on_change=on_any_change)
        st.number_input("Federal pension amount max", step=10.0, key="fed_pension_amount_max", on_change=on_any_change)
        st.number_input("Ontario pension amount max", step=10.0, key="ont_pension_amount_max", on_change=on_any_change)


    cA, cB = st.columns(2)
    if cA.button("Save settings now"):
        save_settings(collect_settings_from_session())
        st.success("Saved.")
    if cB.button("Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        save_settings(DEFAULTS.copy())
        st.success("Reset.")

# ======================
# Helpers
# ======================
from retirement2_helpers import *

# ======================
# Simulation
# ======================
settings_now = collect_settings_from_session()

# Restore UI variables required for charts/metrics
goal_income = float(settings_now["goal_income"])
goal_is_after_tax = bool(settings_now["goal_is_after_tax"])


# Run Simulation
progress_bar = st.progress(0, text="Simulation processing...")

def update_progress(pct: float, label: str):
    progress_bar.progress(pct, text=label)

rows, run_out_age_you = run_model_simulation(settings_now, progress_cb=update_progress)
progress_bar.empty()
df = pd.DataFrame(rows)


# ======================
# Output
# ======================
c1, c2, c3, c4, c5, c6 = st.columns(6)
total_savings_start = float(settings_now["rrif_start_balance"]) + float(settings_now["tfsa_start_balance"]) + float(settings_now["nonreg_start_balance"])
c1.metric("Starting household savings", f"${total_savings_start:,.0f}")
c2.metric("Goal income", f"${goal_income:,.0f}/yr")
c3.metric("Goal type", "After-tax" if goal_is_after_tax else "Pre-tax")
c5.metric("Spouse is", f"{int(st.session_state['spouse_age_diff']):+} years vs you")
c6.metric("Your age when goal first not met", str(run_out_age_you) if run_out_age_you else "Goal met for full model window")

st.subheader("Visuals")

left, right = st.columns(2)

# ======================
# MAIN STACKED INCOME CHART — EXACTLY what you asked for:
# RRIF spend, TFSA spend, CPP, OAS, Pension, RRIF->TFSA transfer, RRIF->NonReg transfer
# Goal line is AFTER-TAX goal (net spend), and transfers do NOT count toward the goal.
# ======================
with left:
    st.write("**Income sources + transfers (stacked) — X axis is YOUR age**")
    fig, ax = plt.subplots()

    cpp_total = df["Your CPP"] + df["Spouse CPP"]
    oas_total = df["Your OAS"] + df["Spouse OAS"]
    pension_total = df["Your Pension (declining)"] + df["Spouse Pension (declining)"]

    rrif_spend = df["RRIF Withdraw (spend)"]
    tfsa_spend = df["TFSA Withdraw (spend)"]
    nonreg_spend = df["NonReg Withdraw (spend)"]

    xfer_tfsa = df["RRIF -> TFSA Transfer"]
    xfer_nonreg = df["RRIF -> NonReg Transfer"]

    bottom = 0
    ax.bar(df["Your Age"], cpp_total, bottom=bottom, label="CPP (total)")
    bottom = bottom + cpp_total

    ax.bar(df["Your Age"], oas_total, bottom=bottom, label="OAS (total)")
    bottom = bottom + oas_total

    ax.bar(df["Your Age"], pension_total, bottom=bottom, label="Pension (total, declining)")
    bottom = bottom + pension_total

    ax.bar(df["Your Age"], rrif_spend, bottom=bottom, label="RRIF withdraw (spend)")
    bottom = bottom + rrif_spend

    ax.bar(df["Your Age"], tfsa_spend, bottom=bottom, label="TFSA withdraw (spend)")
    bottom = bottom + tfsa_spend

    ax.bar(df["Your Age"], nonreg_spend, bottom=bottom, label="NonReg withdraw (spend)")
    bottom = bottom + nonreg_spend

    ax.bar(df["Your Age"], xfer_tfsa, bottom=bottom, label="RRIF → TFSA transfer")
    bottom = bottom + xfer_tfsa

    ax.bar(df["Your Age"], xfer_nonreg, bottom=bottom, label="RRIF → NonReg transfer")

    # Goal line AFTER-TAX (net spend target). Transfers are NOT part of the goal.
    if goal_is_after_tax:
        ax.axhline(goal_income, linestyle="--", linewidth=1, label="Goal (after-tax spend)")
    else:
        ax.axhline(goal_income, linestyle="--", linewidth=1, label="Goal (pre-tax)")

    ax.set_ylabel("$/year (today's dollars)")
    ax.set_xlabel("Your Age")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    st.pyplot(fig)

# Savings/Bucket balances
with right:
    st.write("**Bucket balances over time — X axis is YOUR age**")
    fig2, ax2 = plt.subplots()
    ax2.plot(df["Your Age"], df["RRIF End"], label="RRIF")
    ax2.plot(df["Your Age"], df["TFSA End"], label="TFSA")
    ax2.plot(df["Your Age"], df["NonReg End"], label="NonReg")
    ax2.plot(df["Your Age"], df["Total End (buckets)"], label="Total")
    ax2.set_ylabel("$ (today's dollars)")
    ax2.set_xlabel("Your Age")
    ax2.legend()
    st.pyplot(fig2)

# Net vs goal chart
st.subheader("After-tax cash available vs after-tax goal (transfers excluded from goal) + total tax paid")
fig3, ax3 = plt.subplots()

# LEFT axis = after-tax cash
ax3.plot(df["Your Age"], df["After-Tax Cash Available"], label="After-tax cash available")
ax3.axhline(goal_income, linestyle="--", linewidth=1, label="Goal (after-tax spend)")
ax3.set_ylabel("After-tax $/year (today's dollars)")
ax3.set_xlabel("Your Age")

# RIGHT axis = total tax (household)
ax3_tax = ax3.twinx()
ax3_tax.bar(df["Your Age"], df["Total Tax (est)"], alpha=0.25, label="Total tax (you+spouse)")
ax3_tax.set_ylabel("Total tax $/year (today's dollars)")

# --- COMBINE LEGENDS ---
h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax3_tax.get_legend_handles_labels()

ax3.legend(
    h1 + h2,
    l1 + l2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),  # moves legend below chart
    ncol=3,                      # number of columns in legend
    frameon=False                # cleaner look
)

fig3.tight_layout()  # prevents clipping
st.pyplot(fig3)

st.subheader("Year-by-year table")
# Reorder / Filter some columns for the main view to make it cleaner
cols_to_show = [
    "Year",
    "Status", 
    "Your Age", 
    "Spouse Age",
    "RRIF End", 
    "TFSA End", 
    "NonReg End",
    "Marginal Rate (You)", 
    "Marginal Rate (Spouse)",
    "RRIF Withdraw (spend)", 
    "TFSA Withdraw (spend)",
    "NonReg Withdraw (spend)",
    "RRIF -> TFSA Transfer", 
    "RRIF -> NonReg Transfer",
    "RRIF Split % (You)",
    "Hypo Death Tax", "Hypo Net Estate",
    "Estate Tax Paid", "Net Estate Value", # Actual terminals
    "Total Tax (est)", "After-Tax Cash Available"
]
# Mapping if names are slightly different
available_cols = df.columns.tolist()
cols_to_show = [c for c in cols_to_show if c in available_cols]

st.dataframe(df[cols_to_show], use_container_width=True)

st.download_button(
    "Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="retirement_income_bridge_buckets_with_transfers.csv",
    mime="text/csv"
)