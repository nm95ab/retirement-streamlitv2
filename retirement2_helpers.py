import json
import copy
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple, Optional


# ==========================================================
# DEFAULTS (unchanged)
# ==========================================================
DEFAULTS = {
    # ======================
    # Timeline
    # ======================
    "start_year": 2026,
    "years": 35,
    "your_start_age": 60,
    "spouse_start_age": 60,
    "spouse_start_age_override": False,
    "spouse_age_diff": 0,

    "your_death_age": 95,
    "spouse_death_age": 95,

    # ======================
    # Buckets
    # ======================
    "rrif_start_balance": 500000.0,
    "tfsa_start_balance": 50000.0,
    "nonreg_start_balance": 100000.0,

    # TFSA room tracking
    "tfsa_room_start_total": 0.0,              # household total room now
    "tfsa_room_annual_per_person": 7000.0,     # per person annual room (×2)

    # ======================
    # Return
    # ======================
    "real_return_pct": 3.0,
    "withdraw_at_start_of_year": True,
    "allow_surplus_transfers": True,

    # ======================
    # Incomes
    # ======================
    "your_cpp_amount": 10000.0,
    "your_cpp_start_age": 65,
    "your_oas_amount": 8000.0,
    "your_oas_start_age": 65,
    "your_pension_amount": 0.0,
    "your_pension_start_age": 60,
    "your_pension_decline_pct": 3.0,  # NEW (for declining pension)

    "spouse_cpp_amount": 10000.0,
    "spouse_cpp_start_age": 65,
    "spouse_oas_amount": 8000.0,
    "spouse_oas_start_age": 65,
    "spouse_pension_amount": 0.0,
    "spouse_pension_start_age": 60,
    "spouse_pension_decline_pct": 3.0,  # NEW

    # ======================
    # Goal
    # ======================
    "goal_income": 100000.0,          # after-tax goal
    "current_year_goal_income": 100000.0,
    "goal_is_after_tax": True,

    # ======================
    # Tax / NonReg approximation
    # ======================
    "nonreg_withdraw_taxable_pct": 0.0,  # rough cap-gains proxy
    "income_split_tax_optimization": True,
    "enable_tax": True,

    # ----------------------
    # 2026 Brackets (as you are using)
    # ----------------------
    "federal_brackets_json": (
        "[[58523, 0.14], [117045, 0.205], [181440, 0.26], [258482, 0.29], [1000000000, 0.33]]"
    ),
    "ontario_brackets_json": (
        "[[53891, 0.0505], [107784, 0.0915], [150000, 0.1116], [220000, 0.1216], [1000000000, 0.1316]]"
    ),

    # Credits / surtax / clawback (passed into your helper)
    "fed_basic_personal_amount": 16452.0,
    "fed_bpa_min": 14829.0,                   # NEW: Floor for BPA phase-out
    "fed_bpa_phaseout_start": 181440.0,       # NEW: Income where BPA starts to drop
    "fed_bpa_phaseout_end": 258482.0,         # NEW: Income where BPA hits floor

    "ont_basic_personal_amount": 12989.0,
    "fed_age_amount_max": 9208.0,
    "fed_age_amount_thresh": 44325.0,
    "ont_age_amount_max": 6342.0,
    "ont_age_amount_thresh": 47210.0,
    "fed_pension_amount_max": 2000.0,
    "ont_pension_amount_max": 1796.0,
    "ont_surtax_enabled": True,
    "ont_surtax_t1": 5818.0,
    "ont_surtax_t2": 7446.0,
    "ont_health_premium_enabled": True,

    # NEW: Ontario Health Premium piecewise thresholds
    "ont_health_premium_thresholds_json": (
        "[[20000, 0, 0, 0], [36000, 300, 20000, 0.06], [48000, 450, 36000, 0.06], "
        "[72000, 600, 48000, 0.25], [200000, 750, 72000, 0.25], [1000000000, 900, 200000, 0.25]]"
    ),

    "enable_oas_clawback": True,
    "oas_clawback_threshold": 90997.0,
    "oas_clawback_max_est": 10000.0,          # NEW: Estimated max OAS repayment

    # Used by your Streamlit UI (keep present)
    "withdraw_split_to_you_pct": 50.0,
    "treat_rrif_as_eligible_pension_income_at_65": True,

    # Strategy selector (Streamlit uses it)
    "rrif_strategy_mode": "dynamic",
    "rrif_taxable_target_per_person": 0.0,
    "dynamic_max_marginal_rate": 0.54,
    "dynamic_melt_safety_margin": 0.08,

    # NEW: Optimization and Marginal Rate settings
    "marginal_rate_delta": 50.0,              # Step size for marginal rate calc
    "dynamic_mode_target_r": 0.18,            # Baseline target for dynamic search

    # NEW: RRIF min table (CRA standard for 71+)
    "rrif_min_table_json": json.dumps({
        71: 0.0528, 72: 0.0540, 73: 0.0553, 74: 0.0567, 75: 0.0582,
        76: 0.0598, 77: 0.0617, 78: 0.0636, 79: 0.0658, 80: 0.0682,
        81: 0.0708, 82: 0.0738, 83: 0.0771, 84: 0.0808, 85: 0.0851,
        86: 0.0899, 87: 0.0955, 88: 0.1021, 89: 0.1099, 90: 0.1192,
        91: 0.1306, 92: 0.1449, 93: 0.1634, 94: 0.1879
    }),

    # Debug / Dev
    "enable_debug_logging": True,
}


# ==========================================================
# Cache-safety strategy (NO STALE DATA)
#
# - All caches are pure-function caches keyed by the *inputs that matter*.
# - Any change in settings used by the function changes the cache key.
# - We NEVER cache based on object identity (dict id), only on content fingerprints.
# ==========================================================

def _freeze(obj: Any) -> Any:
    """Make nested structures hashable and deterministic."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, tuple):
        return tuple(_freeze(v) for v in obj)
    return obj


def _tax_settings_key(settings: Dict) -> Tuple[Any, ...]:
    """
    Fingerprint ONLY the settings fields that affect tax + marginal calcs + RRIF min + OHP.
    If any of these values change, cache keys change automatically => no stale data.
    """
    keys = (
        # fed/ont bpa/age/pension credit settings
        "fed_basic_personal_amount", "fed_bpa_min", "fed_bpa_phaseout_start", "fed_bpa_phaseout_end",
        "ont_basic_personal_amount",
        "fed_age_amount_max", "fed_age_amount_thresh",
        "ont_age_amount_max", "ont_age_amount_thresh",
        "fed_pension_amount_max", "ont_pension_amount_max",

        # surtax + OHP
        "ont_surtax_enabled", "ont_surtax_t1", "ont_surtax_t2",
        "ont_health_premium_enabled", "ont_health_premium_thresholds_json",

        # OAS clawback
        "enable_oas_clawback", "oas_clawback_threshold", "oas_clawback_max_est",

        # marginal step
        "marginal_rate_delta",

        # RRIF min table
        "rrif_min_table_json",
    )
    snap = {k: settings.get(k) for k in keys}
    return ("tax_settings_v1", _freeze(snap))


def _brackets_key(brackets: Iterable) -> Tuple[Tuple[float, float], ...]:
    """Normalize brackets into a hashable tuple of (top, rate)."""
    out: List[Tuple[float, float]] = []
    for b in brackets:
        try:
            top, rate = b
            out.append((float(top), float(rate)))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return tuple(out)


def _cents(x: float) -> int:
    """Avoid float drift in keys by quantizing to cents."""
    return int(round(float(x) * 100.0))


# ==========================================================
# 1) Helpers & Utility Logic (unchanged)
# ==========================================================

def get_fixed_inc(age: int, start_age: int, amount: float) -> float:
    return float(amount) if age >= int(start_age) else 0.0


def declining_pension(age: int, start_age: int, amount: float, decline_pct: float) -> float:
    """
    Pension declines each year after start_age by decline_pct (not indexed).
    Example: 3% decline means multiply by (0.97)^(years_since_start).
    """
    if age < int(start_age):
        return 0.0
    yrs = age - int(start_age)
    base = float(amount)
    d = float(decline_pct) / 100.0
    return base * ((1.0 - d) ** yrs)


# ==========================================================
# 2) Tax & Income Logic (Engine) with SAFE caching
# ==========================================================

def _normalize_brackets_input(brackets: Any) -> str:
    """
    Convert brackets input into a deterministic JSON string so it can be cached safely.
    Accepts:
      - JSON string
      - Python list/tuple of pairs
      - None/empty -> "[]"
    """
    if brackets is None:
        return "[]"

    # If already a list/tuple, normalize and dump to JSON for a stable cache key
    if isinstance(brackets, (list, tuple)):
        try:
            arr = [[float(a), float(b)] for a, b in brackets]
            arr.sort(key=lambda x: x[0])
            return json.dumps(arr, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return "[]"

    # Otherwise treat as string
    s = str(brackets).strip()
    return s if s else "[]"


@lru_cache(maxsize=64)
def _parse_brackets_cached(brackets_key: str) -> Tuple[Tuple[float, float], ...]:
    """
    brackets_key is always a JSON string produced by _normalize_brackets_input().
    """
    try:
        data = json.loads(brackets_key)
        out: List[Tuple[float, float]] = []
        for top, rate in data:
            out.append((float(top), float(rate)))
        out.sort(key=lambda x: x[0])
        return tuple(out)
    except Exception:
        return tuple()


def parse_brackets(brackets_input: Any) -> list:
    """
    Public helper used everywhere else.
    Returns list of [top, rate].
    """
    key = _normalize_brackets_input(brackets_input)
    return [list(x) for x in _parse_brackets_cached(key)]


def tax_from_brackets(taxable_income: float, brackets: list) -> float:
    """Calculates tax based on progressive brackets."""
    income = max(0.0, float(taxable_income))
    tax = 0.0
    prev_top = 0.0
    for top, rate in brackets:
        if income <= prev_top:
            break
        amt_in_bracket = min(income, top) - prev_top
        if amt_in_bracket > 0:
            tax += amt_in_bracket * rate
        prev_top = top
    return max(0.0, tax)


def calc_age_amount(net_income: float, age: int, settings: dict, prefix: str) -> float:
    """Calculates Federal or Provincial Age Amount tax credit."""
    if age < 65:
        return 0.0

    max_amt = float(settings.get(f"{prefix}_age_amount_max", 0))
    thresh = float(settings.get(f"{prefix}_age_amount_thresh", 0))

    if net_income <= thresh:
        return max_amt

    reduction = 0.15 * (net_income - thresh)
    return max(0.0, max_amt - reduction)


def calc_fed_bpa(net_income: float, settings: dict) -> float:
    """Calculates the Federal Basic Personal Amount (BPA) with phase-out for high earners."""
    ni = max(0.0, float(net_income))
    bpa_max = float(settings.get("fed_basic_personal_amount", 16452.0))
    bpa_min = float(settings.get("fed_bpa_min", 14829.0))

    start = float(settings.get("fed_bpa_phaseout_start", 181440.0))
    end = float(settings.get("fed_bpa_phaseout_end", 258482.0))

    if ni <= start:
        return bpa_max
    if ni >= end:
        return bpa_min

    reduction = (bpa_max - bpa_min) * (ni - start) / (end - start)
    return max(bpa_min, bpa_max - reduction)


def ontario_surtax(basic_ont_tax: float, settings: dict) -> float:
    """Calculates Ontario Surtax."""
    if not settings.get("ont_surtax_enabled", True):
        return 0.0

    bt = max(0.0, basic_ont_tax)
    t1 = float(settings.get("ont_surtax_t1", 0))
    t2 = float(settings.get("ont_surtax_t2", 0))

    surtax = 0.0
    if bt > t1:
        surtax += 0.20 * (bt - t1)
    if bt > t2:
        surtax += 0.36 * (bt - t2)
    return surtax


@lru_cache(maxsize=16)
def _ohp_tiers_cached(ohp_json: str) -> Optional[Tuple[Tuple[float, float, float, float], ...]]:
    try:
        tiers = json.loads(ohp_json)
        parsed = []
        for thresh, total_max, prev_thresh, rate in tiers:
            parsed.append((float(thresh), float(total_max), float(prev_thresh), float(rate)))
        return tuple(parsed)
    except Exception:
        return None


def ontario_health_premium(taxable_income: float, settings: dict = None) -> float:
    """
    Calculates Ontario Health Premium based on tiered thresholds.
    """
    ti = max(0.0, float(taxable_income))

    ohp_json = settings.get("ont_health_premium_thresholds_json") if settings else None
    if ohp_json:
        tiers = _ohp_tiers_cached(str(ohp_json))
        if tiers:
            prev_max = 0.0
            for thresh, total_max, prev_thresh, rate in tiers:
                if ti <= thresh:
                    return min(total_max, prev_max + rate * (ti - prev_thresh))
                prev_max = total_max
            return prev_max

    # Fallbacks
    if ti <= 20000:
        return 0.0
    if ti <= 36000:
        return min(300.0, 0.06 * (ti - 20000))
    if ti <= 48000:
        return min(450.0, 300.0 + 0.06 * (ti - 36000))
    if ti <= 72000:
        return min(600.0, 450.0 + 0.25 * (ti - 48000))
    if ti <= 200000:
        return min(750.0, 600.0 + 0.25 * (ti - 72000))
    return min(900.0, 750.0 + 0.25 * (ti - 200000))


@lru_cache(maxsize=200_000)
def _calc_total_tax_cached(
    taxable_income_cents: int,
    age: int,
    eligible_pension_cents: int,
    settings_key: Tuple[Any, ...],
    fed_b_key: Tuple[Tuple[float, float], ...],
    ont_b_key: Tuple[Tuple[float, float], ...],
) -> float:
    """
    Cached total tax. Safe because:
    - settings_key is a content fingerprint of all tax-relevant settings
    - brackets are included in the key
    - income/eligible_pension are quantized
    """
    ti = max(0.0, taxable_income_cents / 100.0)
    eligible_pension = max(0.0, eligible_pension_cents / 100.0)

    # Rehydrate brackets to lists for existing helpers
    fed_brackets = list(fed_b_key)
    ont_brackets = list(ont_b_key)

    # Rehydrate minimal settings dict from the fingerprint payload
    # (We stored only tax-related keys inside settings_key.)
    _, frozen = settings_key
    settings = dict(frozen)  # frozen is tuple(sorted((k,v)...))

    # 1. Base Tax on Brackets
    fed_tax_base = tax_from_brackets(ti, fed_brackets)
    ont_tax_base = tax_from_brackets(ti, ont_brackets)

    # 2. Credits (Basic + Age + Pension)
    fed_lowest_rate = fed_brackets[0][1] if fed_brackets else 0.15
    ont_lowest_rate = ont_brackets[0][1] if ont_brackets else 0.0505

    fed_bpa = calc_fed_bpa(ti, settings)
    ont_bpa = float(settings.get("ont_basic_personal_amount", 0))

    fed_age = calc_age_amount(ti, age, settings, "fed")
    ont_age = calc_age_amount(ti, age, settings, "ont")

    fed_pen = min(eligible_pension, float(settings.get("fed_pension_amount_max", 0)))
    ont_pen = min(eligible_pension, float(settings.get("ont_pension_amount_max", 0)))

    fed_credits = (fed_bpa + fed_age + fed_pen) * fed_lowest_rate
    ont_credits = (ont_bpa + ont_age + ont_pen) * ont_lowest_rate

    fed_tax = max(0.0, fed_tax_base - fed_credits)
    ont_tax = max(0.0, ont_tax_base - ont_credits)

    # 3. Ontario Specifics
    surtax = ontario_surtax(ont_tax, settings)
    ohp = 0.0
    if settings.get("ont_health_premium_enabled", True):
        ohp = ontario_health_premium(ti, settings)

    # 4. OAS Clawback
    oas_clawback = 0.0
    if settings.get("enable_oas_clawback", True) and age >= 65:
        thresh = float(settings.get("oas_clawback_threshold", 90997.0))
        if ti > thresh:
            repay = 0.15 * (ti - thresh)
            oas_cap = float(settings.get("oas_clawback_max_est", 10000.0))
            oas_clawback = min(repay, oas_cap)

    return float(fed_tax + ont_tax + surtax + ohp + oas_clawback)


def calc_total_tax(
    taxable_income: float,
    age: int,
    settings: dict,
    fed_brackets: list,
    ont_brackets: list,
    eligible_pension: float = 0.0
) -> float:
    """
    Public API unchanged.
    Internally uses safe cache keyed by tax-relevant settings + brackets + (income, age, eligible_pension).
    """
    sk = _tax_settings_key(settings)
    fbk = _brackets_key(fed_brackets)
    obk = _brackets_key(ont_brackets)
    return _calc_total_tax_cached(_cents(taxable_income), int(age), _cents(eligible_pension), sk, fbk, obk)


@lru_cache(maxsize=200_000)
def _calc_marginal_rate_cached(
    income_cents: int,
    age: int,
    eligible_pension_cents: int,
    delta_cents: int,
    settings_key: Tuple[Any, ...],
    fed_b_key: Tuple[Tuple[float, float], ...],
    ont_b_key: Tuple[Tuple[float, float], ...],
) -> float:
    inc = income_cents / 100.0
    delta = max(0.01, delta_cents / 100.0)
    elig = eligible_pension_cents / 100.0
    t1 = _calc_total_tax_cached(income_cents, age, eligible_pension_cents, settings_key, fed_b_key, ont_b_key)
    t2 = _calc_total_tax_cached(_cents(inc + delta), age, eligible_pension_cents, settings_key, fed_b_key, ont_b_key)
    return float((t2 - t1) / delta)


def calc_marginal_rate(
    current_income: float,
    age: int,
    settings: dict,
    fed_brackets: list,
    ont_brackets: list,
    eligible_pension: float = 0.0,
    delta: float = None
) -> float:
    """
    Public API unchanged. Internally cached, safely.
    """
    if delta is None:
        delta = float(settings.get("marginal_rate_delta", 50.0))
    sk = _tax_settings_key(settings)
    fbk = _brackets_key(fed_brackets)
    obk = _brackets_key(ont_brackets)
    return _calc_marginal_rate_cached(
        _cents(current_income),
        int(age),
        _cents(eligible_pension),
        _cents(delta),
        sk,
        fbk,
        obk,
    )


# ==========================================================
# 3) RRIF & Optimization Logic with SAFE caching
# ==========================================================

@lru_cache(maxsize=8)
def _rrif_table_cached(rrif_min_table_json: str) -> Optional[Dict[int, float]]:
    try:
        table = json.loads(rrif_min_table_json)
        return {int(k): float(v) for k, v in table.items()}
    except Exception:
        return None


def get_rrif_min_percentage(age: int, settings: dict = None) -> float:
    """Returns CRA minimum withdrawal factor."""
    if age < 55:
        return 0.0
    if age < 71:
        return 1.0 / (90.0 - age)

    table_json = settings.get("rrif_min_table_json") if settings else None
    if table_json:
        table = _rrif_table_cached(str(table_json))
        if table:
            return float(table.get(age, 0.20))

    # Fallback hardcoded table
    table2 = {
        71: 0.0528, 72: 0.0540, 73: 0.0553, 74: 0.0567, 75: 0.0582,
        76: 0.0598, 77: 0.0617, 78: 0.0636, 79: 0.0658, 80: 0.0682,
        81: 0.0708, 82: 0.0738, 83: 0.0771, 84: 0.0808, 85: 0.0851,
        86: 0.0899, 87: 0.0955, 88: 0.1021, 89: 0.1099, 90: 0.1192,
        91: 0.1306, 92: 0.1449, 93: 0.1634, 94: 0.1879
    }
    return float(table2.get(age, 0.20))


@lru_cache(maxsize=50_000)
def _find_income_matching_target_rate_cached(
    target_rate_milli: int,
    age: int,
    eligible_pension_cents: int,
    settings_key: Tuple[Any, ...],
    fed_b_key: Tuple[Tuple[float, float], ...],
    ont_b_key: Tuple[Tuple[float, float], ...],
) -> float:
    target_rate = target_rate_milli / 1000.0
    eligible_pension = eligible_pension_cents / 100.0

    fed_brackets = list(fed_b_key)
    ont_brackets = list(ont_b_key)

    points = set([0.0])
    for top, _rate in fed_brackets:
        points.add(float(top))
    for top, _rate in ont_brackets:
        points.add(float(top))
    points.add(90997.0)

    sorted_points = sorted([p for p in points if p < 2_000_000.0])
    if not sorted_points:
        return 0.0

    top_start = sorted_points[-1]
    top_rate = _calc_marginal_rate_cached(
        _cents(top_start + 1000.0),
        age,
        _cents(eligible_pension),
        _cents(50.0),  # delta will be pulled from settings inside public call, but here we need a value
        settings_key,
        fed_b_key,
        ont_b_key,
    )

    if target_rate >= (top_rate - 0.005):
        return float(top_start)

    for p in sorted_points:
        rate = _calc_marginal_rate_cached(
            _cents(p + 50.0),
            age,
            _cents(eligible_pension),
            _cents(50.0),
            settings_key,
            fed_b_key,
            ont_b_key,
        )
        if rate >= (target_rate - 0.005):
            return float(p)

    return float(sorted_points[-1])


def find_income_matching_target_rate(
    target_rate: float,
    age: int,
    settings: dict,
    fed_brackets: list,
    ont_brackets: list,
    eligible_pension: float = 0.0
) -> float:
    """
    Public API unchanged. Internally cached safely using tax-relevant settings + brackets.
    """
    sk = _tax_settings_key(settings)
    fbk = _brackets_key(fed_brackets)
    obk = _brackets_key(ont_brackets)
    # milli-rate avoids float drift in cache key
    return _find_income_matching_target_rate_cached(
        int(round(float(target_rate) * 1000.0)),
        int(age),
        _cents(eligible_pension),
        sk,
        fbk,
        obk,
    )


@lru_cache(maxsize=10_000)
def _project_future_peak_rate_cached(
    start_year_idx: int,
    years_max: int,
    rrif0_cents: int,
    growth_rate_milli: int,
    ages_you: int,
    ages_spouse: int,
    death_y: int,
    death_s: int,
    settings_key: Tuple[Any, ...],
    fed_b_key: Tuple[Tuple[float, float], ...],
    ont_b_key: Tuple[Tuple[float, float], ...],
    # income timing / pension timing keys (must be included to prevent stale)
    timing_key: Tuple[Any, ...],
) -> float:
    """
    Cached peak-rate computation. Safe because:
    - settings_key contains all tax knobs (incl OAS/OHP/credits/marginal step/RRIF table)
    - timing_key includes CPP/OAS/pension start ages/amounts/declines that affect future fixed taxable income
    """
    rrif0 = rrif0_cents / 100.0
    if rrif0 <= 1.0:
        return 0.0

    g = growth_rate_milli / 1000.0

    fed_b = list(fed_b_key)
    ont_b = list(ont_b_key)

    # Rehydrate minimal settings dicts:
    _, frozen_tax = settings_key
    tax_settings = dict(frozen_tax)

    # timing_key was frozen dict (tuple(sorted(...)))
    _, frozen_timing = timing_key
    timing = dict(frozen_timing)

    treat_rrif_eligible = bool(timing.get("treat_rrif_as_eligible_pension_income_at_65", True))

    def fixed_taxable_for_person(age: int, who: str) -> Tuple[float, float]:
        if who == "you":
            cpp = get_fixed_inc(age, timing["your_cpp_start_age"], timing["your_cpp_amount"])
            oas = get_fixed_inc(age, timing["your_oas_start_age"], timing["your_oas_amount"])
            pen = declining_pension(
                age,
                timing["your_pension_start_age"],
                timing["your_pension_amount"],
                timing.get("your_pension_decline_pct", 0.0),
            )
        else:
            cpp = get_fixed_inc(age, timing["spouse_cpp_start_age"], timing["spouse_cpp_amount"])
            oas = get_fixed_inc(age, timing["spouse_oas_start_age"], timing["spouse_oas_amount"])
            pen = declining_pension(
                age,
                timing["spouse_pension_start_age"],
                timing["spouse_pension_amount"],
                timing.get("spouse_pension_decline_pct", 0.0),
            )
        fixed_taxable = cpp + oas + pen
        eligible_pension = pen
        return float(fixed_taxable), float(eligible_pension)

    # Build years list
    first_k = start_year_idx + 1
    last_k = years_max - 1

    years = []
    for k in range(first_k, last_k + 1):
        ay = ages_you + k
        aS = ages_spouse + k
        alive_y = ay <= death_y
        alive_s = aS <= death_s
        if not alive_y and not alive_s:
            break
        years.append((k, ay, aS, alive_y, alive_s))

    if not years:
        age_final = max(death_y, death_s) + 1
        # Use cached marginal rate with the correct settings/brackets
        return float(
            _calc_marginal_rate_cached(
                _cents(rrif0),
                int(age_final),
                0,
                _cents(float(tax_settings.get("marginal_rate_delta", 50.0))),
                settings_key,
                fed_b_key,
                ont_b_key,
            )
        )

    def max_income_for_rate(R: float, age: int, eligible_pension: float) -> float:
        if R <= 0.0:
            return 0.0

        delta = float(tax_settings.get("marginal_rate_delta", 50.0))

        lo = 0.0
        hi = 50_000.0
        for _ in range(30):
            r_hi = _calc_marginal_rate_cached(
                _cents(hi),
                int(age),
                _cents(eligible_pension),
                _cents(delta),
                settings_key,
                fed_b_key,
                ont_b_key,
            )
            if r_hi > R + 1e-4:
                break
            hi *= 1.6
            if hi > 5_000_000:
                break

        r_hi = _calc_marginal_rate_cached(
            _cents(hi),
            int(age),
            _cents(eligible_pension),
            _cents(delta),
            settings_key,
            fed_b_key,
            ont_b_key,
        )
        if r_hi <= R + 1e-4:
            return float(hi)

        for _ in range(40):
            mid = (lo + hi) / 2.0
            r_mid = _calc_marginal_rate_cached(
                _cents(mid),
                int(age),
                _cents(eligible_pension),
                _cents(delta),
                settings_key,
                fed_b_key,
                ont_b_key,
            )
            if r_mid <= R + 1e-4:
                lo = mid
            else:
                hi = mid
        return float(lo)

    def feasible(R: float) -> bool:
        caps: List[float] = []

        for (_k, age_y, age_s, alive_y, alive_s) in years:
            if alive_y:
                fixed_y, elig_y = fixed_taxable_for_person(age_y, "you")
            else:
                fixed_y, elig_y = 0.0, 0.0

            if alive_s:
                fixed_s, elig_s = fixed_taxable_for_person(age_s, "spouse")
            else:
                fixed_s, elig_s = 0.0, 0.0

            if alive_y and alive_s:
                cap_inc_y = max_income_for_rate(R, age_y, elig_y)
                cap_inc_s = max_income_for_rate(R, age_s, elig_s)
                room_y = max(0.0, cap_inc_y - fixed_y)
                room_s = max(0.0, cap_inc_s - fixed_s)
                cap_total = room_y + room_s
            else:
                if alive_y:
                    cap_inc = max_income_for_rate(R, age_y, elig_y)
                    cap_total = max(0.0, cap_inc - fixed_y)
                else:
                    cap_inc = max_income_for_rate(R, age_s, elig_s)
                    cap_total = max(0.0, cap_inc - fixed_s)

            caps.append(float(cap_total))

        max_next = 0.0
        for i in range(len(caps) - 1, -1, -1):
            max_start = caps[i] + (max_next / (1.0 + g)) if (1.0 + g) > 1e-12 else (caps[i] + max_next)
            max_next = max_start
        max_start_first_year = max_next

        # RRIF minimum check (uses timing+tax settings; RRIF table is included in settings_key)
        bal_bound = rrif0
        for idx, (_k, age_y, age_s, alive_y, alive_s) in enumerate(years):
            age_for_min = age_y if alive_y else age_s
            # settings dict for RRIF min: needs rrif_min_table_json
            min_pct = get_rrif_min_percentage(age_for_min, {**tax_settings})
            min_w = bal_bound * min_pct
            if caps[idx] + 1e-6 < min_w:
                return False
            bal_bound = (bal_bound - min_w) * (1.0 + g)

        return rrif0 <= max_start_first_year + 1e-6

    lo = 0.0
    hi = 0.90
    for _ in range(12):
        if feasible(hi):
            break
        hi = min(0.99, hi + 0.03)
    if not feasible(hi):
        return float(hi)

    for _ in range(35):
        mid = (lo + hi) / 2.0
        if feasible(mid):
            hi = mid
        else:
            lo = mid

    return float(hi)


def project_future_peak_rate(
    start_year_idx: int,
    years_max: int,
    settings: dict,
    current_rrif_bal: float,
    fed_b: list,
    ont_b: list,
    ages: dict,
    growth_rate: float,
) -> float:
    """
    Public API unchanged. Internally cached safely.
    """
    # tax knobs fingerprint
    sk = _tax_settings_key(settings)
    fbk = _brackets_key(fed_b)
    obk = _brackets_key(ont_b)

    # timing knobs fingerprint (anything that affects the fixed income streams)
    timing_keys = (
        "your_cpp_amount", "your_cpp_start_age",
        "your_oas_amount", "your_oas_start_age",
        "your_pension_amount", "your_pension_start_age", "your_pension_decline_pct",
        "spouse_cpp_amount", "spouse_cpp_start_age",
        "spouse_oas_amount", "spouse_oas_start_age",
        "spouse_pension_amount", "spouse_pension_start_age", "spouse_pension_decline_pct",
        "treat_rrif_as_eligible_pension_income_at_65",
    )
    timing_snap = {k: settings.get(k) for k in timing_keys}
    timing_key = ("timing_v1", _freeze(timing_snap))

    death_y = int(settings.get("your_death_age", 95))
    death_s = int(settings.get("spouse_death_age", 95))

    return _project_future_peak_rate_cached(
        int(start_year_idx),
        int(years_max),
        _cents(current_rrif_bal),
        int(round(float(growth_rate) * 1000.0)),
        int(ages["you"]),
        int(ages["spouse"]),
        death_y,
        death_s,
        sk,
        fbk,
        obk,
        timing_key,
    )