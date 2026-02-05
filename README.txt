========================================================================
           Retirement Income Bridge & Tax Optimization Simulator
========================================================================

Overview
--------
This program is a robust retirement planning tool designed for Canadian 
households. it simulates income sources, draws from various investment 
vehicles (RRIF, TFSA, Non-Reg), and implements advanced tax optimization 
strategies like "RRIF Melting" and "Tax Smoothing" to maximize after-tax 
wealth and minimize estate taxes.

Key Features
------------
- Household Simulation: Models two spouses with individual life expectancies.
- Automatic Tax Splitting: Simulates income splitting (e.g., RRIF income).
- Tax Smoothing (Levelized Melt): Optimizes RRIF withdrawals to maintain a 
  consistent tax bracket throughout retirement.
- Multi-Account Support: Handles RRIF (taxable), TFSA (tax-free), and 
  Non-Registered (taxable growth) accounts.
- OAS Clawback Modeling: Optional modeling of Old Age Security clawbacks.
- Interactive Dashboard: Streamlit-based UI for real-time scenario testing.

File Structure
--------------
- retirement2.py: The main Streamlit application (UI/Frontend).
- retirement_simulation.py: The core logic engine (Backend).
- retirement2_helpers.py: Tax tables, defaults, and utility functions.
- retirement_settings.json: Local storage for your simulation settings.

How to Run
----------
1. Ensure you have Python installed.
2. Install dependencies:
   pip install streamlit pandas matplotlib
3. Run the application:
   streamlit run retirement2.py

Usage Tips
----------
- Use the sidebar to adjust timeline, assets, income (CPP/OAS), and goals.
- The "Levelized Melt" mode will attempt to empty your RRIF systematically
  to avoid a massive tax bill at the end of the simulation.
- You can save and load your custom scenarios using the sidebar buttons.

------------------------------------------------------------------------
Created for optimization of retirement wealth and estate tax management.
