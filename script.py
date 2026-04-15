"""
A-CAES Dispatch Optimization
Wind Farm + Adiabatic Compressed Air Energy Storage
Optimization: Always charge when price < 0 €/MWh (rule-based fix)
             + LP optimization for remaining hours
"""

import pandas as pd
import numpy as np
import pulp
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_FILE = Path(__file__).resolve().parent / "Wind_farm_data.xlsx"
OUTPUT_FILE = Path(__file__).resolve().parent / "ACAES_Optimized_Dispatch.csv"

# ─────────────────────────────────────────────
# 1. SYSTEM PARAMETERS (from Excel)
# ─────────────────────────────────────────────
DISCHARGE_POWER_MW   = 50        # MW
CHARGE_POWER_MW      = 50        # MW
STORAGE_CAPACITY_MWh = 500       # MWh
ROUNDTRIP_EFF        = 0.70      # 70%
CHARGE_EFF           = np.sqrt(ROUNDTRIP_EFF)   # ~0.8367
DISCHARGE_EFF        = np.sqrt(ROUNDTRIP_EFF)   # ~0.8367

# Turbine
N_TURBINES           = 33
RATED_POWER_KW       = 3000
CUT_IN               = 3.5       # m/s
RATED_WIND           = 16.5      # m/s
CUT_OFF              = 25.0      # m/s

# Economics
CAPEX_ACAES_EUR_KW   = 1600      # €/kW
CAPEX_WIND_EUR_KW    = 1200      # €/kW
TOTAL_WIND_MW        = N_TURBINES * RATED_POWER_KW / 1000   # 99 MW
OM_ACAES_PCT         = 0.02      # 2% of CAPEX/year
OM_WIND_PCT          = 0.015     # 1.5% of CAPEX/year
DISCOUNT_RATE        = 0.06
PROJECT_LIFETIME     = 30        # years

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}.\n"
        "Place Wind_farm_data.xlsx next to script.py or update DATA_FILE."
    )

# Spot prices (15-min → hourly)
df_spot = pd.read_excel(
    DATA_FILE,
    sheet_name='Spot', index_col=0, header=0
).iloc[1:]
df_spot.index = pd.to_datetime(df_spot.index, dayfirst=True, errors='coerce')
df_spot = df_spot[df_spot.index.notna()]
df_spot['SE4 [€]'] = pd.to_numeric(df_spot['SE4 [€]'], errors='coerce')
prices = df_spot['SE4 [€]'].resample('h').first().ffill()

# Wind speeds (hourly)
df_wind = pd.read_excel(
    DATA_FILE,
    sheet_name='Hourly Wind Data', skiprows=3, index_col=0
)
df_wind.index = pd.to_datetime(df_wind.index, errors='coerce')
df_wind = df_wind[df_wind.index.notna()]
df_wind.columns = ['wind_speed']
df_wind['wind_speed'] = pd.to_numeric(df_wind['wind_speed'], errors='coerce').fillna(0)

# Align on same index
idx = prices.index.intersection(df_wind.index)
prices    = prices.loc[idx]
wind_speed = df_wind.loc[idx, 'wind_speed']
T = len(idx)
print(f"  Hours: {T}  |  Date range: {idx[0].date()} → {idx[-1].date()}")
print(f"  Negative price hours: {(prices < 0).sum()}")

# ─────────────────────────────────────────────
# 3. WIND POWER CALCULATION
# ─────────────────────────────────────────────
def wind_power_mw(ws):
    """Simple piecewise linear power curve for Vestas V90/3000."""
    ws = np.array(ws, dtype=float)
    p  = np.zeros_like(ws)
    # Cut-in to rated: cubic interpolation approximation
    mask_ramp = (ws >= CUT_IN) & (ws < RATED_WIND)
    p[mask_ramp] = RATED_POWER_KW * ((ws[mask_ramp] - CUT_IN) / (RATED_WIND - CUT_IN)) ** 3
    # Rated to cut-off
    mask_rated = (ws >= RATED_WIND) & (ws < CUT_OFF)
    p[mask_rated] = RATED_POWER_KW
    # Scale to full farm (kW → MW)
    return p * N_TURBINES / 1000

wind_mw = wind_power_mw(wind_speed.values)

print(f"  Avg wind production: {wind_mw.mean():.1f} MW  |  "
      f"Total annual: {wind_mw.sum():.0f} MWh")

# ─────────────────────────────────────────────
# 4. ORIGINAL DISPATCH (from CSV reference)
#    Replicate the simple rule: charge if price < median, discharge if price > 75th pct
# ─────────────────────────────────────────────
p = prices.values.copy()
p_median  = np.nanmedian(p)
p75       = np.nanpercentile(p, 75)

orig_charge    = np.zeros(T)
orig_discharge = np.zeros(T)
orig_soc       = np.zeros(T)
soc = 0.0

for t in range(T):
    price = p[t]
    if np.isnan(price):
        orig_soc[t] = soc
        continue
    if price <= p_median and soc < STORAGE_CAPACITY_MWh:
        c = min(CHARGE_POWER_MW, (STORAGE_CAPACITY_MWh - soc) / CHARGE_EFF)
        orig_charge[t] = c
        soc = min(STORAGE_CAPACITY_MWh, soc + c * CHARGE_EFF)
    elif price >= p75 and soc > 0:
        d = min(DISCHARGE_POWER_MW, soc * DISCHARGE_EFF)
        orig_discharge[t] = d
        soc = max(0, soc - d / DISCHARGE_EFF)
    orig_soc[t] = soc

# ─────────────────────────────────────────────
# 5. OPTIMIZED DISPATCH — Rule: ALWAYS charge when price < 0
#    + LP optimization for all remaining decisions
# ─────────────────────────────────────────────
print("\nRunning LP optimization (this may take ~1-2 min)...")

prob = pulp.LpProblem("ACAES_Optimized", pulp.LpMaximize)

# Decision variables
charge_var    = [pulp.LpVariable(f"c_{t}", 0, CHARGE_POWER_MW)    for t in range(T)]
discharge_var = [pulp.LpVariable(f"d_{t}", 0, DISCHARGE_POWER_MW) for t in range(T)]
soc_var       = [pulp.LpVariable(f"s_{t}", 0, STORAGE_CAPACITY_MWh) for t in range(T)]
# Binary: can't charge and discharge simultaneously
b_charge = [pulp.LpVariable(f"bc_{t}", cat='Binary') for t in range(T)]

# Objective: maximize net revenue from CAES arbitrage
prob += pulp.lpSum(
    discharge_var[t] * (p[t] if not np.isnan(p[t]) else 0) -
    charge_var[t]    * (p[t] if not np.isnan(p[t]) else 0)
    for t in range(T)
)

# Constraints
for t in range(T):
    price = p[t] if not np.isnan(p[t]) else 0

    # SoC continuity
    if t == 0:
        prob += soc_var[t] == charge_var[t] * CHARGE_EFF - discharge_var[t] / DISCHARGE_EFF
    else:
        prob += soc_var[t] == soc_var[t-1] + charge_var[t] * CHARGE_EFF - discharge_var[t] / DISCHARGE_EFF

    # Mutual exclusion via binary
    prob += charge_var[t]    <= CHARGE_POWER_MW    * b_charge[t]
    prob += discharge_var[t] <= DISCHARGE_POWER_MW * (1 - b_charge[t])

    # RULE: Avoid discharge when price < 0; charging is allowed if storage capacity exists
    if price < 0:
        prob += b_charge[t] == 1
        prob += discharge_var[t] == 0

# End SoC ≥ 0 (already handled by variable bounds)
prob += soc_var[T-1] >= 0

# Solve
solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)
prob.solve(solver)
print(f"  Solver status: {pulp.LpStatus[prob.status]}")

# Extract results
opt_charge    = np.array([pulp.value(charge_var[t])    or 0 for t in range(T)])
opt_discharge = np.array([pulp.value(discharge_var[t]) or 0 for t in range(T)])
opt_soc       = np.array([pulp.value(soc_var[t])       or 0 for t in range(T)])

# ─────────────────────────────────────────────
# 6. ECONOMICS
# ─────────────────────────────────────────────
def compute_economics(charge_arr, discharge_arr, label):
    price_arr = np.nan_to_num(p, nan=0.0)

    # CAES arbitrage revenue
    discharge_rev = np.sum(discharge_arr * price_arr)
    charge_cost   = np.sum(charge_arr    * price_arr)
    net_caes      = discharge_rev - charge_cost

    # Wind-to-grid: whatever wind isn't used for charging goes to grid
    # Wind used for charging = charge power (drawn from wind first)
    wind_to_caes  = np.minimum(charge_arr, wind_mw)
    wind_to_grid  = np.maximum(wind_mw - charge_arr, 0)
    wind_rev      = np.sum(wind_to_grid * price_arr)
    total_rev     = wind_rev + discharge_rev - charge_cost

    return {
        'label':            label,
        'discharge_rev':    discharge_rev,
        'charge_cost':      charge_cost,
        'net_caes_arb':     net_caes,
        'wind_rev':         wind_rev,
        'total_annual_rev': total_rev,
        'charge_hours':     int((charge_arr > 0).sum()),
        'discharge_hours':  int((discharge_arr > 0).sum()),
        'idle_hours':       int(T - (charge_arr > 0).sum() - (discharge_arr > 0).sum()),
    }

orig_eco = compute_economics(orig_charge,    orig_discharge,    "Original")
opt_eco  = compute_economics(opt_charge, opt_discharge, "Optimized")

# CAPEX
total_acaes_mw      = DISCHARGE_POWER_MW
capex_acaes         = CAPEX_ACAES_EUR_KW * total_acaes_mw * 1000
capex_wind          = CAPEX_WIND_EUR_KW  * TOTAL_WIND_MW  * 1000
total_capex         = capex_acaes + capex_wind

om_acaes_annual     = capex_acaes * OM_ACAES_PCT
om_wind_annual      = capex_wind  * OM_WIND_PCT
total_om_annual     = om_acaes_annual + om_wind_annual

# NPV & Annuity factor
annuity = (1 - (1 + DISCOUNT_RATE)**(-PROJECT_LIFETIME)) / DISCOUNT_RATE

def npv_metrics(eco):
    annual_net = eco['total_annual_rev'] - total_om_annual
    npv = -total_capex + annual_net * annuity
    # Simple payback (undiscounted)
    payback = total_capex / annual_net if annual_net > 0 else float('inf')
    lcoe_wind = (capex_wind + om_wind_annual * annuity) / (wind_mw.sum() * annuity)
    return {
        'annual_net_revenue': annual_net,
        'NPV':                npv,
        'payback_years':      payback,
        'LCOE_wind_EUR_MWh':  lcoe_wind,
    }

orig_npv = npv_metrics(orig_eco)
opt_npv  = npv_metrics(opt_eco)

# ─────────────────────────────────────────────
# 7. PRINT RESULTS
# ─────────────────────────────────────────────
def fmt(v):
    if abs(v) >= 1e6:  return f"€{v/1e6:,.2f}M"
    if abs(v) >= 1e3:  return f"€{v/1e3:,.1f}K"
    return f"€{v:,.2f}"

print("\n" + "="*62)
print("  A-CAES + WIND FARM — ECONOMIC RESULTS")
print("="*62)

print("\n── CAPEX ────────────────────────────────────────────────")
print(f"  A-CAES CAPEX  ({DISCHARGE_POWER_MW} MW × {CAPEX_ACAES_EUR_KW} €/kW):  {fmt(capex_acaes)}")
print(f"  Wind CAPEX    ({TOTAL_WIND_MW:.0f} MW × {CAPEX_WIND_EUR_KW} €/kW):  {fmt(capex_wind)}")
print(f"  TOTAL CAPEX:                             {fmt(total_capex)}")

print("\n── ANNUAL O&M ───────────────────────────────────────────")
print(f"  A-CAES O&M  ({OM_ACAES_PCT*100:.0f}% of CAPEX):        {fmt(om_acaes_annual)}")
print(f"  Wind O&M    ({OM_WIND_PCT*100:.1f}% of CAPEX):      {fmt(om_wind_annual)}")
print(f"  TOTAL O&M:                               {fmt(total_om_annual)}")

for eco, npv_r in [(orig_eco, orig_npv), (opt_eco, opt_npv)]:
    tag = eco['label'].upper()
    print(f"\n── ANNUAL REVENUE — {tag} ──────────────────────────")
    print(f"  Wind-to-grid revenue:        {fmt(eco['wind_rev'])}")
    print(f"  CAES discharge revenue:      {fmt(eco['discharge_rev'])}")
    print(f"  CAES charging cost:         -{fmt(eco['charge_cost'])}")
    print(f"  Net CAES arbitrage:          {fmt(eco['net_caes_arb'])}")
    print(f"  TOTAL annual revenue:        {fmt(eco['total_annual_rev'])}")
    print(f"  Net (after O&M):             {fmt(npv_r['annual_net_revenue'])}")
    print(f"  NPV ({PROJECT_LIFETIME}yr, {DISCOUNT_RATE*100:.0f}% disc.):         {fmt(npv_r['NPV'])}")
    pb = npv_r['payback_years']
    print(f"  Simple payback:              {pb:.1f} years" if pb < 999 else "  Simple payback:              Never")
    print(f"  Dispatch — Charge: {eco['charge_hours']}h | "
          f"Discharge: {eco['discharge_hours']}h | Idle: {eco['idle_hours']}h")

delta_rev = opt_eco['total_annual_rev'] - orig_eco['total_annual_rev']
delta_npv = opt_npv['NPV'] - orig_npv['NPV']
print(f"\n── IMPROVEMENT (Optimized vs Original) ─────────────────")
print(f"  Additional annual revenue:   {fmt(delta_rev)}")
print(f"  Additional NPV:              {fmt(delta_npv)}")
print("="*62)

# ─────────────────────────────────────────────
# 8. SAVE RESULTS TO CSV
# ─────────────────────────────────────────────
results_df = pd.DataFrame({
    'Timestamp':          idx,
    'Spot_Price_EUR_MWh': p,
    'Wind_Production_MW': wind_mw,
    'Orig_Charge_MW':     orig_charge,
    'Orig_Discharge_MW':  orig_discharge,
    'Orig_SoC_MWh':       orig_soc,
    'Opt_Charge_MW':      opt_charge,
    'Opt_Discharge_MW':   opt_discharge,
    'Opt_SoC_MWh':        opt_soc,
})
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nDispatch results saved to {OUTPUT_FILE.name}")