import copy
import logging
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

from retirement2_helpers import (
    parse_brackets,
    calc_total_tax,
    calc_marginal_rate,
    get_rrif_min_percentage,
    find_income_matching_target_rate,
    project_future_peak_rate,
    get_fixed_inc,
    declining_pension,
    DEFAULTS,
)


@dataclass
class Balances:
    rrif: float
    tfsa: float
    nonreg: float


@dataclass
class PotBalances:
    rrif_pot: float
    tfsa_pot: float
    nonreg_pot: float


@dataclass
class Incomes:
    fy_cpp: float
    fy_oas: float
    fy_pen: float
    fs_cpp: float
    fs_oas: float
    fs_pen: float


@dataclass
class CashResult:
    gross_total: float
    net_total: float
    tax_total: float
    rrif_you: float
    rrif_sp: float
    marginal_y: float = 0.0
    marginal_sp: float = 0.0


@dataclass
class WithdrawPlan:
    # spending withdrawals
    w_rrif_spend: float = 0.0
    w_nonreg_spend: float = 0.0
    w_tfsa_spend: float = 0.0

    # extra RRIF withdrawal to "fill bracket" and then transfer
    w_rrif_transfer_gross: float = 0.0

    # after-tax transfers (contributions)
    rrif_to_tfsa: float = 0.0
    rrif_to_nonreg: float = 0.0

    # reporting split RRIF amounts
    rrif_you: float = 0.0
    rrif_sp: float = 0.0

    # reporting totals
    gross_total: float = 0.0
    net_total: float = 0.0
    tax_total: float = 0.0
    marginal_y: float = 0.0
    marginal_sp: float = 0.0

    # melt gating diagnostics
    allow_melt: bool = True
    peak_future_rate: float = 0.0
    current_household_marginal: float = 0.0
    melt_safety_margin: float = 0.0


class RetirementSimulator:
    def __init__(self, settings_input: Dict, progress_cb: Optional[Callable[[float, str], None]] = None):
        self.progress_cb = progress_cb
        # Merge settings first
        self.s = self._merge_settings(settings_input)

        # logging setup (must exist before optional global optimization)
        self.logger = logging.getLogger("RetirementSimulator")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)
        if self.s.get("enable_debug_logging"):
            self.logger.setLevel(logging.DEBUG)

        self._opt_completed = False
        # NEW: Replace the existing "dynamic optimization" by selecting the best dynamic knobs
        # via a lifetime objective (sum after-tax spending used + after-tax estate at second death).
        # This does NOT affect the fixed strategy mode at all.
        self._maybe_optimize_dynamic_policy_globally()

        # basic run control
        self.years = int(self.s["years"])
        self.start_year = int(self.s["start_year"])

        # bracket tables
        self.fed_b = parse_brackets(self.s["federal_brackets_json"])
        self.ont_b = parse_brackets(self.s["ontario_brackets_json"])

        # returns / toggles
        self.real_ret = float(self.s["real_return_pct"]) / 100.0

        # non-reg "after-tax" return approximation (lower than RRIF/TFSA return)
        # If omitted, defaults to the same as real_return_pct.
        self.nonreg_real_ret = float(self.s.get("nonreg_real_return_pct", self.s["real_return_pct"])) / 100.0

        # You set this to 0.0 because you want to assume no taxable non-reg withdrawal income (cash/no gains).
        self.nonreg_taxable_pct = 0.0

        self.goal_is_after_tax = bool(self.s.get("goal_is_after_tax", True))
        self.withdraw_at_start = bool(self.s.get("withdraw_at_start_of_year", True))

        # dynamic melt gating safety margin (absolute, e.g. 0.08 = 8 points)
        # NOTE: may have been tuned by global optimization above.
        self.dynamic_melt_safety_margin = float(self.s.get("dynamic_melt_safety_margin", 0.08))

        # TFSA room tracking
        self.tfsa_room = float(self.s.get("tfsa_room_start_total", 0.0))
        self.tfsa_room_annual = float(self.s.get("tfsa_room_annual_per_person", 0.0)) * 2.0
        self.next_year_recontribution = 0.0

        # starting balances
        self.bal = Balances(
            rrif=float(self.s["rrif_start_balance"]),
            tfsa=float(self.s["tfsa_start_balance"]),
            nonreg=float(self.s["nonreg_start_balance"]),
        )

        # spouse start age logic must match Streamlit
        self.spouse_start_age = self._resolve_spouse_start_age()

        # outputs
        self.rows: List[Dict] = []
        self.run_out_age_you: Optional[int] = None

    # -------------------------
    # Setup / Settings helpers
    # -------------------------
    def _merge_settings(self, settings_input: Dict) -> Dict:
        s = copy.deepcopy(DEFAULTS)
        if isinstance(settings_input, dict):
            s.update(settings_input)

        # Ensure defaults exist even if DEFAULTS doesn't have them
        s.setdefault("dynamic_melt_safety_margin", 0.08)

        # NEW: global dynamic optimizer defaults (only relevant in dynamic mode)
        s.setdefault("dynamic_global_opt_enabled", True)
        s.setdefault("dynamic_global_opt_estate_weight", 1.0)
        # Optional grids (user can override by passing lists)
        # s.setdefault("dynamic_global_opt_target_r_grid", [0.14, 0.16, 0.18, 0.20, 0.22])
        # s.setdefault("dynamic_global_opt_safety_grid", [0.00, 0.04, 0.08, 0.12])

        return s

    def _resolve_spouse_start_age(self) -> int:
        if bool(self.s.get("spouse_start_age_override", False)):
            return int(self.s["spouse_start_age"])
        spouse_age = int(self.s["your_start_age"]) + int(self.s.get("spouse_age_diff", 0))
        self.s["spouse_start_age"] = spouse_age  # ensure consistent
        return spouse_age

    # ---------------------------------------------------
    # NEW: Global dynamic policy optimization (lifetime)
    # ---------------------------------------------------
    def _score_rows_lifetime_objective(self, rows: List[Dict], estate_weight: float) -> float:
        """
        Objective = sum(after-tax spending actually used each year) + estate_weight * after-tax estate (second death)
        IMPORTANT: We do NOT count transfers as "spending" to avoid double-counting.
        """
        total_spend = 0.0
        terminal_estate: Optional[float] = None

        for r in rows:
            status = r.get("Status", "")
            if status == "Terminal":
                terminal_estate = float(r.get("Net Estate Value", 0.0))
                break

            goal_after_tax = float(r.get("Goal (after-tax)", 0.0))
            cash_after_tax = float(r.get("After-Tax Cash Available", 0.0))

            # Spending used should not exceed cash available.
            # If goal can't be met, spend what's available.
            spend_used = min(cash_after_tax, goal_after_tax)
            total_spend += spend_used

        # If sim ended before Terminal, use last row's hypothetical estate snapshot
        if terminal_estate is None and rows:
            terminal_estate = float(rows[-1].get("Hypo Net Estate", 0.0))

        return total_spend + float(estate_weight) * float(terminal_estate or 0.0)

    def _run_sim_for_candidate_settings(self, candidate_settings: Dict) -> Tuple[List[Dict], Optional[int]]:
        """
        Run a full simulation using candidate settings, but skip re-optimizing
        to avoid recursion.
        """
        s2 = dict(candidate_settings)
        s2["_skip_dynamic_global_opt"] = True
        sim = RetirementSimulator(s2)
        return sim.run()

    def _maybe_optimize_dynamic_policy_globally(self) -> None:
        """
        If mode is dynamic AND global opt is enabled, choose best:
          - dynamic_mode_target_r
          - dynamic_melt_safety_margin
        by maximizing lifetime objective.
        """
        mode = self.s.get("rrif_strategy_mode", "fixed")
        if mode != "dynamic":
            return

        if bool(self.s.get("_skip_dynamic_global_opt", False)):
            return

        if not bool(self.s.get("dynamic_global_opt_enabled", True)):
            return

        estate_weight = float(self.s.get("dynamic_global_opt_estate_weight", 1.0))

        default_target_r_grid = [0.14, 0.16, 0.18, 0.20, 0.22]
        default_safety_grid = [0.00, 0.04, 0.08, 0.12]

        target_r_grid = self.s.get("dynamic_global_opt_target_r_grid", default_target_r_grid)
        safety_grid = self.s.get("dynamic_global_opt_safety_grid", default_safety_grid)

        best_score = -1e30
        best_target_r = float(self.s.get("dynamic_mode_target_r", 0.18))
        best_safety = float(self.s.get("dynamic_melt_safety_margin", 0.08))

        base = dict(self.s)
        total_perms = len(target_r_grid) * len(safety_grid)
        curr_idx = 0

        for tr in target_r_grid:
            for sm in safety_grid:
                curr_idx += 1
                if self.progress_cb:
                    pct = 0.9 * (curr_idx / total_perms)
                    self.progress_cb(pct, f"Optimizing strategy... {curr_idx}/{total_perms}")

                cand = dict(base)
                cand["dynamic_mode_target_r"] = float(tr)
                cand["dynamic_melt_safety_margin"] = float(sm)

                rows, _ = self._run_sim_for_candidate_settings(cand)
                score = self._score_rows_lifetime_objective(rows, estate_weight=estate_weight)

                if score > best_score:
                    best_score = score
                    best_target_r = float(tr)
                    best_safety = float(sm)

        # Apply best knobs to this simulator's settings
        self.s["dynamic_mode_target_r"] = best_target_r
        self.s["dynamic_melt_safety_margin"] = best_safety
        self._opt_completed = True

        self.logger.info(
            f"[Dynamic Global Opt] Selected dynamic_mode_target_r={best_target_r:.2%}, "
            f"dynamic_melt_safety_margin={best_safety:.2%}, "
            f"objective_score={best_score:,.2f}"
        )

    # -------------------------
    # Tax & RRIF helpers (Optimized with Caching)
    # -------------------------
    def _tax_key(self, taxable_income: float, age: int, eligible_pension: float) -> Tuple[int, int, int]:
        """Rounds dollars to improve cache hit rate."""
        return (int(round(taxable_income)), int(age), int(round(eligible_pension)))

    @lru_cache(maxsize=200000)
    def _tax_cached(self, taxable_income_i: int, age: int, eligible_pension_i: int) -> float:
        """Call the tax engine and cache the result."""
        return float(
            calc_total_tax(
                float(taxable_income_i),
                age,
                self.s,
                self.fed_b,
                self.ont_b,
                float(eligible_pension_i),
            )
        )

    def _first_bracket_ceiling_per_person(self) -> float:
        fed_first = float(self.fed_b[0][0]) if self.fed_b else 58523.0
        ont_first = float(self.ont_b[0][0]) if self.ont_b else 49231.0
        return min(fed_first, ont_first)

    def _rrif_mandatory_min(self, age_y: int, rrif_balance: float) -> float:
        pct = get_rrif_min_percentage(age_y, self.s)
        return max(0.0, rrif_balance * pct)

    def _marginal_rate(self, taxable_income: float, age: int, eligible_pension: float) -> float:
        """Computes the marginal tax rate for a single person using a configurable step."""
        step = float(self.s.get("marginal_rate_delta", 100.0))
        k0 = self._tax_key(taxable_income, age, eligible_pension)
        k1 = self._tax_key(taxable_income + step, age, eligible_pension)
        t0 = self._tax_cached(*k0)
        t1 = self._tax_cached(*k1)
        return (t1 - t0) / step

    def _compute_household_cash(
        self,
        age_y: int,
        age_s: int,
        inc: Incomes,
        w_rrif_total_gross: float,
        w_nonreg_gross: float,
        w_tfsa_gross: float,
        alive_y: bool = True,
        alive_s: bool = True,
    ) -> CashResult:
        """
        - RRIF split is searched to MINIMIZE total tax whenever both alive.
        - NonReg withdrawal is treated as 100% principal (no taxable component),
          because self.nonreg_taxable_pct = 0.0.
        """
        enable_tax = bool(self.s.get("enable_tax", True))

        # RRIF income allocation
        if alive_y and alive_s:
            total = w_rrif_total_gross
            best_tax = None
            best_you = total * 0.5
            best_sp = total * 0.5

            treat_rrif_eligible = bool(self.s.get("treat_rrif_as_eligible_pension_income_at_65", True))

            taxable_nonreg_you = 0.0
            taxable_nonreg_sp = 0.0

            def evaluate_split(y_amt: float) -> Tuple[float, float, float]:
                s_amt = total - y_amt
                inc_y = inc.fy_cpp + inc.fy_oas + inc.fy_pen + y_amt + taxable_nonreg_you
                inc_s = inc.fs_cpp + inc.fs_oas + inc.fs_pen + s_amt + taxable_nonreg_sp
                elig_y = inc.fy_pen + (y_amt if (age_y >= 65 and treat_rrif_eligible) else 0.0)
                elig_s = inc.fs_pen + (s_amt if (age_s >= 65 and treat_rrif_eligible) else 0.0)
                
                ky = self._tax_key(inc_y, age_y, elig_y)
                ks = self._tax_key(inc_s, age_s, elig_s)
                
                t_y = self._tax_cached(*ky)
                t_s = self._tax_cached(*ks)
                return (t_y + t_s), y_amt, s_amt

            # Pass 1: Coarse scan (100 steps)
            num_steps = 100
            for i in range(num_steps + 1):
                y_curr = (total * i) / num_steps
                tot_tax, y_res, s_res = evaluate_split(y_curr)
                if best_tax is None or tot_tax < best_tax - 1e-7:
                    best_tax, best_you, best_sp = tot_tax, y_res, s_res

            # Pass 2: Fine refinement (+/- 1 coarse step)
            if total > 0:
                c_step = total / num_steps
                low = max(0.0, best_you - c_step)
                high = min(total, best_you + c_step)
                for i in range(num_steps + 1):
                    y_curr = low + (high - low) * i / num_steps
                    tot_tax, y_res, s_res = evaluate_split(y_curr)
                    if tot_tax < best_tax - 1e-7:
                        best_tax, best_you, best_sp = tot_tax, y_res, s_res

            rrif_you = best_you
            rrif_sp = best_sp
        elif alive_y:
            rrif_you = w_rrif_total_gross
            rrif_sp = 0.0
        elif alive_s:
            rrif_you = 0.0
            rrif_sp = w_rrif_total_gross
        else:
            rrif_you = w_rrif_total_gross
            rrif_sp = 0.0

        taxable_nonreg_you = 0.0
        taxable_nonreg_sp = 0.0

        inc_y_taxable = inc.fy_cpp + inc.fy_oas + inc.fy_pen + rrif_you + taxable_nonreg_you
        inc_s_taxable = inc.fs_cpp + inc.fs_oas + inc.fs_pen + rrif_sp + taxable_nonreg_sp

        if not enable_tax:
            total_tax = 0.0
            marg_y = 0.0
            marg_sp = 0.0
        else:
            treat_rrif_eligible = bool(self.s.get("treat_rrif_as_eligible_pension_income_at_65", True))

            tax_y = 0.0
            marg_y = 0.0
            if alive_y:
                elig_y = inc.fy_pen + (rrif_you if (age_y >= 65 and treat_rrif_eligible) else 0.0)
                ky = self._tax_key(inc_y_taxable, age_y, elig_y)
                tax_y = self._tax_cached(*ky)
                marg_y = self._marginal_rate(inc_y_taxable, age_y, elig_y)

            tax_sp = 0.0
            marg_sp = 0.0
            if alive_s:
                elig_s = inc.fs_pen + (rrif_sp if (age_s >= 65 and treat_rrif_eligible) else 0.0)
                ks = self._tax_key(inc_s_taxable, age_s, elig_s)
                tax_sp = self._tax_cached(*ks)
                marg_sp = self._marginal_rate(inc_s_taxable, age_s, elig_s)

            total_tax = tax_y + tax_sp

        nonreg_nontaxable = w_nonreg_gross * (1.0 - self.nonreg_taxable_pct)

        gross_total = (inc_y_taxable + inc_s_taxable) + w_tfsa_gross + nonreg_nontaxable
        net_total = gross_total - total_tax

        return CashResult(
            gross_total=gross_total,
            net_total=net_total,
            tax_total=total_tax,
            rrif_you=rrif_you,
            rrif_sp=rrif_sp,
            marginal_y=marg_y,
            marginal_sp=marg_sp,
        )

    def _metric_cash(self, gross_cash: float, net_cash: float) -> float:
        return net_cash if self.goal_is_after_tax else gross_cash

    # -------------------------
    # Solvers (binary search)
    # -------------------------
    def _solve_rrif_additional_to_reach_goal(
        self,
        age_y: int,
        age_s: int,
        inc: Incomes,
        base_rrif: float,
        base_nonreg: float,
        base_tfsa: float,
        target_goal: float,
        cap_additional_rrif: float,
        alive_y: bool = True,
        alive_s: bool = True,
    ) -> float:
        c0 = self._compute_household_cash(age_y, age_s, inc, base_rrif, base_nonreg, base_tfsa, alive_y, alive_s)
        if self._metric_cash(c0.gross_total, c0.net_total) >= target_goal or cap_additional_rrif <= 0:
            return 0.0

        lo, hi = 0.0, cap_additional_rrif
        for _ in range(20):
            mid = (lo + hi) / 2.0
            cm = self._compute_household_cash(
                age_y, age_s, inc, base_rrif + mid, base_nonreg, base_tfsa, alive_y, alive_s
            )
            if self._metric_cash(cm.gross_total, cm.net_total) >= target_goal:
                hi = mid
            else:
                lo = mid
        return hi

    def _solve_nonreg_additional_to_reach_goal(
        self,
        age_y: int,
        age_s: int,
        inc: Incomes,
        base_rrif: float,
        base_nonreg: float,
        base_tfsa: float,
        target_goal: float,
        cap_nonreg: float,
        alive_y: bool = True,
        alive_s: bool = True,
    ) -> float:
        c0 = self._compute_household_cash(age_y, age_s, inc, base_rrif, base_nonreg, base_tfsa, alive_y, alive_s)
        if self._metric_cash(c0.gross_total, c0.net_total) >= target_goal or cap_nonreg <= 0:
            return 0.0

        lo, hi = 0.0, cap_nonreg
        for _ in range(20):
            mid = (lo + hi) / 2.0
            cm = self._compute_household_cash(
                age_y, age_s, inc, base_rrif, base_nonreg + mid, base_tfsa, alive_y, alive_s
            )
            if self._metric_cash(cm.gross_total, cm.net_total) >= target_goal:
                hi = mid
            else:
                lo = mid
        return hi

    def _solve_melt_amount_under_marginal_cap(
        self,
        age_y: int,
        age_s: int,
        inc: Incomes,
        base_plan: WithdrawPlan,
        pots: PotBalances,
        bracket_room_household: float,
        peak_future_rate: float,
        alive_y: bool = True,
        alive_s: bool = True,
    ) -> float:
        """
        Binary search for the maximum RRIF melt amount that keeps the household marginal 
        tax rate below the peak_future_rate (minus small epsilon).
        """
        lo = 0.0
        hi = max(0.0, bracket_room_household - base_plan.w_rrif_spend)
        hi = min(hi, max(0.0, pots.rrif_pot - base_plan.w_rrif_spend))
        
        if hi <= 0:
            return 0.0

        best_amt = 0.0
        # We want marginal + safety < peak_future_rate. 
        # But here peak_future_rate is the target we compare AGAINST.
        # The gating logic in _plan_withdrawals already checked 
        # (current_marg + safety) < peak_future_rate.
        # Now we just want to ensure that even after melting, we don't exceed the peak rate.
        
        for _ in range(15):
            mid = (lo + hi) / 2.0
            test_cash = self._compute_household_cash(
                age_y, age_s, inc, 
                base_plan.w_rrif_spend + mid, 
                base_plan.w_nonreg_spend, 
                base_plan.w_tfsa_spend, 
                alive_y, alive_s
            )
            marg = max(test_cash.marginal_y, test_cash.marginal_sp)
            
            # Use a tiny buffer to avoid edge-of-bracket flip-flopping
            if marg < peak_future_rate - 0.0001:
                best_amt = mid
                lo = mid
            else:
                hi = mid
        
        return best_amt


    # -------------------------
    # Per-year calculations
    # -------------------------
    def _ages_for_year_index(self, i: int) -> Tuple[int, int]:
        return int(self.s["your_start_age"]) + i, int(self.spouse_start_age) + i

    def _alive_flags(self, age_y: int, age_s: int) -> Tuple[bool, bool]:
        return age_y <= int(self.s["your_death_age"]), age_s <= int(self.s["spouse_death_age"])

    def _last_survivor_flags(self) -> Tuple[bool, bool]:
        dy = int(self.s["your_death_age"])
        ds = int(self.s["spouse_death_age"])
        if dy > ds:
            return True, False
        if ds > dy:
            return False, True
        return True, False

    def _update_tfsa_room_for_new_year(self, i: int) -> None:
        self.tfsa_room += self.tfsa_room_annual
        if i > 0:
            self.tfsa_room += self.next_year_recontribution

    def _pots_for_year(self) -> PotBalances:
        if self.withdraw_at_start:
            return PotBalances(self.bal.rrif, self.bal.tfsa, self.bal.nonreg)

        return PotBalances(
            self.bal.rrif * (1.0 + self.real_ret),
            self.bal.tfsa * (1.0 + self.real_ret),
            self.bal.nonreg * (1.0 + self.nonreg_real_ret),
        )

    def _fixed_incomes_for_year(self, age_y: int, age_s: int, alive_y: bool, alive_s: bool) -> Incomes:
        fy_cpp = get_fixed_inc(age_y, self.s["your_cpp_start_age"], self.s["your_cpp_amount"]) if alive_y else 0.0
        fy_oas = get_fixed_inc(age_y, self.s["your_oas_start_age"], self.s["your_oas_amount"]) if alive_y else 0.0
        fy_pen = (
            declining_pension(
                age_y,
                self.s["your_pension_start_age"],
                self.s["your_pension_amount"],
                self.s.get("your_pension_decline_pct", 0.0),
            )
            if alive_y
            else 0.0
        )

        fs_cpp = get_fixed_inc(age_s, self.s["spouse_cpp_start_age"], self.s["spouse_cpp_amount"]) if alive_s else 0.0
        fs_oas = get_fixed_inc(age_s, self.s["spouse_oas_start_age"], self.s["spouse_oas_amount"]) if alive_s else 0.0
        fs_pen = (
            declining_pension(
                age_s,
                self.s["spouse_pension_start_age"],
                self.s["spouse_pension_amount"],
                self.s.get("spouse_pension_decline_pct", 0.0),
            )
            if alive_s
            else 0.0
        )

        return Incomes(fy_cpp, fy_oas, fy_pen, fs_cpp, fs_oas, fs_pen)

    def _goal_for_year_index(self, i: int) -> float:
        return float(self.s.get("current_year_goal_income", self.s["goal_income"])) if i == 0 else float(
            self.s["goal_income"]
        )

    def _bracket_room_household(
        self, inc: Incomes, current_year_idx: int, alive_y: bool, alive_s: bool
    ) -> Tuple[float, float, float]:
        """
        Returns:
          (bracket_ceiling_per_person, household_room, peak_future_rate)

        peak_future_rate is only meaningful in dynamic mode; otherwise 0.0.
        """
        mode = self.s.get("rrif_strategy_mode", "fixed")
        peak_r = 0.0

        if mode == "dynamic":
            ages = {"you": int(self.s["your_start_age"]), "spouse": int(self.spouse_start_age)}

            peak_r = project_future_peak_rate(
                current_year_idx,
                self.years,
                self.s,
                self.bal.rrif,
                self.fed_b,
                self.ont_b,
                ages,
                self.real_ret,
            )

            target_r = float(self.s.get("dynamic_mode_target_r", 0.18))

            age_curr_y = int(self.s["your_start_age"]) + current_year_idx
            age_curr_s = int(self.spouse_start_age) + current_year_idx
            ceiling_age = max(age_curr_y if alive_y else 0, age_curr_s if alive_s else 0)
            if ceiling_age < 55:
                ceiling_age = 65

            bracket_ceiling = find_income_matching_target_rate(
                target_r, ceiling_age, self.s, self.fed_b, self.ont_b, inc.fy_pen + inc.fs_pen
            )
        else:
            # FIXED MODE: unchanged
            custom_target = float(self.s.get("rrif_taxable_target_per_person", 0.0))
            bracket_ceiling = custom_target if custom_target > 0 else self._first_bracket_ceiling_per_person()

        base_taxable_y = inc.fy_cpp + inc.fy_oas + inc.fy_pen
        base_taxable_s = inc.fs_cpp + inc.fs_oas + inc.fs_pen

        room_y = max(0.0, bracket_ceiling - base_taxable_y) if alive_y else 0.0
        room_s = max(0.0, bracket_ceiling - base_taxable_s) if alive_s else 0.0

        return bracket_ceiling, room_y + room_s, peak_r

    # -------------------------
    # Withdrawal strategy
    # Spend order: NonReg -> RRIF -> TFSA
    # Transfer rule: RRIF -> TFSA always (if room)
    # RRIF -> NonReg only if age >= 75 AND TFSA room is 0
    # -------------------------
    def _plan_withdrawals_nonreg_then_rrif_then_tfsa_then_bracket_fill(
        self,
        age_y: int,
        age_s: int,
        inc: Incomes,
        pots: PotBalances,
        goal: float,
        bracket_room_household: float,
        mandatory_min: float,
        peak_future_rate: float,
        alive_y: bool = True,
        alive_s: bool = True,
    ) -> WithdrawPlan:
        plan = WithdrawPlan()
        plan.peak_future_rate = float(peak_future_rate)
        plan.melt_safety_margin = float(self.dynamic_melt_safety_margin)

        self.logger.debug(f"Planning withdrawals for Goal: ${goal:,.2f} ...")

        # Step A: mandatory RRIF (cannot avoid)
        plan.w_rrif_spend = min(pots.rrif_pot, mandatory_min)
        if plan.w_rrif_spend > 0:
            self.logger.debug(f"Step A: Withdrawing mandatory RRIF min: ${plan.w_rrif_spend:,.2f}")

        cash0 = self._compute_household_cash(age_y, age_s, inc, plan.w_rrif_spend, 0.0, 0.0, alive_y, alive_s)
        plan.gross_total, plan.net_total, plan.tax_total = cash0.gross_total, cash0.net_total, cash0.tax_total
        plan.rrif_you, plan.rrif_sp = cash0.rrif_you, cash0.rrif_sp
        plan.marginal_y, plan.marginal_sp = cash0.marginal_y, cash0.marginal_sp
        metric0 = self._metric_cash(plan.gross_total, plan.net_total)

        # Step B: NonReg FIRST to meet goal
        if metric0 < goal and pots.nonreg_pot > 0:
            add_nonreg = self._solve_nonreg_additional_to_reach_goal(
                age_y,
                age_s,
                inc,
                base_rrif=plan.w_rrif_spend,
                base_nonreg=plan.w_nonreg_spend,
                base_tfsa=plan.w_tfsa_spend,
                target_goal=goal,
                cap_nonreg=pots.nonreg_pot,
                alive_y=alive_y,
                alive_s=alive_s,
            )
            plan.w_nonreg_spend += min(add_nonreg, pots.nonreg_pot)
            self.logger.debug(f"Step B: Tapping Non-Reg for shortfall: ${plan.w_nonreg_spend:,.2f}")

            cash1 = self._compute_household_cash(
                age_y, age_s, inc, plan.w_rrif_spend, plan.w_nonreg_spend, plan.w_tfsa_spend, alive_y, alive_s
            )
            plan.gross_total, plan.net_total, plan.tax_total = cash1.gross_total, cash1.net_total, cash1.tax_total
            plan.rrif_you, plan.rrif_sp = cash1.rrif_you, cash1.rrif_sp
            plan.marginal_y, plan.marginal_sp = cash1.marginal_y, cash1.marginal_sp
            metric0 = self._metric_cash(plan.gross_total, plan.net_total)

        # Step C: RRIF next if still short
        if metric0 < goal:
            cap_extra_rrif = max(0.0, pots.rrif_pot - plan.w_rrif_spend)
            if cap_extra_rrif > 0:
                add_rrif = self._solve_rrif_additional_to_reach_goal(
                    age_y,
                    age_s,
                    inc,
                    base_rrif=plan.w_rrif_spend,
                    base_nonreg=plan.w_nonreg_spend,
                    base_tfsa=plan.w_tfsa_spend,
                    target_goal=goal,
                    cap_additional_rrif=cap_extra_rrif,
                    alive_y=alive_y,
                    alive_s=alive_s,
                )
                plan.w_rrif_spend += add_rrif
                self.logger.debug(f"Step C: Still short, pulling extra RRIF: ${add_rrif:,.2f}")

                cash2 = self._compute_household_cash(
                    age_y, age_s, inc, plan.w_rrif_spend, plan.w_nonreg_spend, plan.w_tfsa_spend, alive_y, alive_s
                )
                plan.gross_total, plan.net_total, plan.tax_total = cash2.gross_total, cash2.net_total, cash2.tax_total
                plan.rrif_you, plan.rrif_sp = cash2.rrif_you, cash2.rrif_sp
                plan.marginal_y, plan.marginal_sp = cash2.marginal_y, cash2.marginal_sp
                metric0 = self._metric_cash(plan.gross_total, plan.net_total)

        # Step D: TFSA last
        if metric0 < goal and pots.tfsa_pot > 0:
            shortfall = goal - metric0
            plan.w_tfsa_spend += min(shortfall, pots.tfsa_pot)
            self.logger.debug(f"Step D: Final shortfall, tapping TFSA: ${plan.w_tfsa_spend:,.2f}")

            cash3 = self._compute_household_cash(
                age_y, age_s, inc, plan.w_rrif_spend, plan.w_nonreg_spend, plan.w_tfsa_spend, alive_y, alive_s
            )
            plan.gross_total, plan.net_total, plan.tax_total = cash3.gross_total, cash3.net_total, cash3.tax_total
            plan.rrif_you, plan.rrif_sp = cash3.rrif_you, cash3.rrif_sp
            plan.marginal_y, plan.marginal_sp = cash3.marginal_y, cash3.marginal_sp
            metric0 = self._metric_cash(plan.gross_total, plan.net_total)

        # Step E: bracket-fill RRIF + transfer incremental after-tax cash
        if bool(self.s.get("allow_surplus_transfers", True)):
            # 1) Determine the "Net Cash at Goal" to identify true excess from fixed income
            if self.goal_is_after_tax:
                net_at_goal = goal
            else:
                net_at_goal = (
                    plan.net_total * (min(1.0, goal / plan.gross_total))
                    if plan.gross_total > 1e-6
                    else 0.0
                )

            # 2) Capture baseline surplus (fixed income > goal) BEFORE melt
            pre_melt_surplus = max(0.0, plan.net_total - net_at_goal)
            if pre_melt_surplus > 0:
                to_tfsa = min(pre_melt_surplus, self.tfsa_room)
                to_nonreg = pre_melt_surplus - to_tfsa
                plan.rrif_to_tfsa += to_tfsa
                plan.rrif_to_nonreg += to_nonreg
                self.tfsa_room -= to_tfsa

            # 3) Decide whether to allow melt (dynamic gating)
            mode = self.s.get("rrif_strategy_mode", "fixed")
            current_household_marg = max(plan.marginal_y, plan.marginal_sp)
            plan.current_household_marginal = float(current_household_marg)

            allow_melt = True
            if mode == "dynamic" and peak_future_rate > 0:
                allow_melt = (current_household_marg + self.dynamic_melt_safety_margin) < peak_future_rate
                self.logger.info(
                    f"Melt gating (Dynamic): Current Marg {current_household_marg:.1%} + "
                    f"Safety {self.dynamic_melt_safety_margin:.1%} vs Peak {peak_future_rate:.1%} "
                    f"-> Allow={allow_melt}"
                )

            plan.allow_melt = bool(allow_melt)

            # 4) Apply Melt (Bracket Fill) only if allowed
            if allow_melt:
                net_before_melt = plan.net_total

                # remaining_room is household room minus what we already withdrew from RRIF for spending
                remaining_room = max(0.0, bracket_room_household - plan.w_rrif_spend)
                remaining_room = min(remaining_room, max(0.0, pots.rrif_pot - plan.w_rrif_spend))

                if remaining_room > 0:
                    mode = self.s.get("rrif_strategy_mode", "fixed")

                    melt_amt = remaining_room
                    if mode == "dynamic" and peak_future_rate > 0:
                        melt_amt = self._solve_melt_amount_under_marginal_cap(
                            age_y=age_y,
                            age_s=age_s,
                            inc=inc,
                            base_plan=plan,
                            pots=pots,
                            bracket_room_household=bracket_room_household,
                            peak_future_rate=peak_future_rate,
                            alive_y=alive_y,
                            alive_s=alive_s,
                        )

                    plan.w_rrif_transfer_gross = float(melt_amt)

                    if plan.w_rrif_transfer_gross > 0:
                        net_before_melt = plan.net_total

                        cash_with_transfer = self._compute_household_cash(
                            age_y,
                            age_s,
                            inc,
                            plan.w_rrif_spend + plan.w_rrif_transfer_gross,
                            plan.w_nonreg_spend,
                            plan.w_tfsa_spend,
                            alive_y,
                            alive_s,
                        )

                        plan.gross_total = cash_with_transfer.gross_total
                        plan.net_total = cash_with_transfer.net_total
                        plan.tax_total = cash_with_transfer.tax_total
                        plan.rrif_you = cash_with_transfer.rrif_you
                        plan.rrif_sp = cash_with_transfer.rrif_sp
                        plan.marginal_y = cash_with_transfer.marginal_y
                        plan.marginal_sp = cash_with_transfer.marginal_sp
                        plan.current_household_marginal = float(max(plan.marginal_y, plan.marginal_sp))

                        incremental_after_tax = max(0.0, plan.net_total - net_before_melt)

                        if incremental_after_tax > 0:
                            to_tfsa = min(incremental_after_tax, self.tfsa_room)
                            leftover = incremental_after_tax - to_tfsa

                            plan.rrif_to_tfsa += to_tfsa
                            self.tfsa_room -= to_tfsa

                            plan.rrif_to_nonreg += leftover

            return plan

    # -------------------------
    # Apply withdrawals + growth
    # -------------------------
    def _apply_plan_to_balances(self, pots: PotBalances, plan: WithdrawPlan) -> Tuple[Balances, float]:
        total_rrif_withdraw_gross = plan.w_rrif_spend + plan.w_rrif_transfer_gross

        rrif_after = max(0.0, pots.rrif_pot - total_rrif_withdraw_gross)
        tfsa_after = max(0.0, pots.tfsa_pot - plan.w_tfsa_spend)
        nonreg_after = max(0.0, pots.nonreg_pot - plan.w_nonreg_spend)

        tfsa_after += plan.rrif_to_tfsa
        nonreg_after += plan.rrif_to_nonreg

        if self.withdraw_at_start:
            rrif_end = rrif_after * (1.0 + self.real_ret)
            tfsa_end = tfsa_after * (1.0 + self.real_ret)
            nonreg_end = nonreg_after * (1.0 + self.nonreg_real_ret)
        else:
            rrif_end = rrif_after
            tfsa_end = tfsa_after
            nonreg_end = nonreg_after

        next_year_recontribution = plan.w_tfsa_spend
        return Balances(rrif_end, tfsa_end, nonreg_end), next_year_recontribution

    def _record_row(
        self,
        year: int,
        age_y: int,
        age_s: int,
        inc: Incomes,
        goal: float,
        bracket_ceiling: float,
        bracket_room_household: float,
        mandatory_min: float,
        pots_start: Balances,
        plan: WithdrawPlan,
        bal_end: Balances,
        alive_y: bool = True,
        alive_s: bool = True,
        estate_tax: float = 0.0,
        net_estate: float = 0.0,
        peak_future_rate: float = 0.0,
    ) -> None:
        if not alive_y and not alive_s:
            status = "Terminal"
        elif alive_y and alive_s:
            status = "Couple"
        else:
            status = "Survivor"

        # Calculate what the estate tax WOULD BE if both died today
        pots = PotBalances(pots_start.rrif, pots_start.tfsa, pots_start.nonreg)
        liquidation_gross = pots.rrif_pot

        hypo_inc = Incomes(0, 0, 0, 0, 0, 0)
        lsy, lss = self._last_survivor_flags()
        hypo_result = self._compute_household_cash(age_y, age_s, hypo_inc, liquidation_gross, 0, 0, lsy, lss)
        hypo_tax = hypo_result.tax_total

        current_net_estate = pots.tfsa_pot + pots.nonreg_pot + (liquidation_gross - hypo_tax)

        self.rows.append(
            {
                "Year": year,
                "Your Age": age_y,
                "Spouse Age": age_s,
                "Status": status,
                "RRIF Start": float(pots_start.rrif),
                "TFSA Start": float(pots_start.tfsa),
                "NonReg Start": float(pots_start.nonreg),
                "Your CPP": float(inc.fy_cpp),
                "Your OAS": float(inc.fy_oas),
                "Your Pension (declining)": float(inc.fy_pen),
                "Spouse CPP": float(inc.fs_cpp),
                "Spouse OAS": float(inc.fs_oas),
                "Spouse Pension (declining)": float(inc.fs_pen),
                "RRIF Withdraw (spend)": float(plan.w_rrif_spend),
                "TFSA Withdraw (spend)": float(plan.w_tfsa_spend),
                "NonReg Withdraw (spend)": float(plan.w_nonreg_spend),
                "RRIF Withdraw (You)": float(plan.rrif_you),
                "RRIF Withdraw (Spouse)": float(plan.rrif_sp),
                "RRIF -> TFSA Transfer": float(plan.rrif_to_tfsa),
                "RRIF -> NonReg Transfer": float(plan.rrif_to_nonreg),
                "After-Tax Cash Available": float(plan.net_total),
                "Total Tax (est)": float(plan.tax_total),
                "Marginal Rate (You)": float(plan.marginal_y),
                "Marginal Rate (Spouse)": float(plan.marginal_sp),
                "RRIF End": float(bal_end.rrif),
                "TFSA End": float(bal_end.tfsa),
                "NonReg End": float(bal_end.nonreg),
                "Total End (buckets)": float(bal_end.rrif + bal_end.tfsa + bal_end.nonreg),
                "Bracket Ceiling (per person)": float(bracket_ceiling),
                "Bracket Room Household": float(bracket_room_household),
                "RRIF Mandatory Minimum": float(mandatory_min),
                "RRIF Withdraw (transfer gross)": float(plan.w_rrif_transfer_gross),
                "TFSA Room End": float(self.tfsa_room),
                "Goal (after-tax)": float(goal),
                "Goal Is After Tax": bool(self.goal_is_after_tax),
                "Gross Cash Available": float(plan.gross_total),
                "Estate Tax Paid": float(estate_tax) if (status == "Terminal") else 0.0,
                "Net Estate Value": float(net_estate) if (status == "Terminal") else 0.0,
                "Hypo Death Tax": float(hypo_tax),
                "Hypo Net Estate": float(current_net_estate),
                "RRIF Split % (You)": float((plan.rrif_you / (plan.rrif_you + plan.rrif_sp)) * 100.0)
                if (plan.rrif_you + plan.rrif_sp) > 1e-4
                else 0.0,
                "NonReg Real Return %": float(self.nonreg_real_ret * 100.0),
                "Dynamic Peak Future Rate": float(peak_future_rate),
                "Current Household Marginal": float(plan.current_household_marginal),
                "Dynamic Melt Safety Margin": float(plan.melt_safety_margin),
                "Melt Allowed": bool(plan.allow_melt),
            }
        )

    # -------------------------
    # Main run
    # -------------------------
    def run(self) -> Tuple[List[Dict], Optional[int]]:
        for i in range(self.years):
            if self.progress_cb:
                if self._opt_completed:
                    pct = 0.9 + 0.1 * (i / self.years)
                else:
                    pct = i / self.years
                self.progress_cb(pct, f"Running simulation... Year {self.start_year + i}")

            year = self.start_year + i
            age_y, age_s = self._ages_for_year_index(i)

            self.logger.info(f"--- Year {i}: {year} (Ages: You={age_y}, Spouse={age_s}) ---")

            alive_y, alive_s = self._alive_flags(age_y, age_s)
            if not alive_y and not alive_s:
                inc = Incomes(0, 0, 0, 0, 0, 0)
                pots = self._pots_for_year()
                liquidation_gross = pots.rrif_pot

                lsy, lss = self._last_survivor_flags()
                last_result = self._compute_household_cash(age_y, age_s, inc, liquidation_gross, 0, 0, lsy, lss)
                tax_paid = last_result.tax_total
                net_estate = pots.tfsa_pot + pots.nonreg_pot + (liquidation_gross - tax_paid)

                terminal_plan = WithdrawPlan(
                    w_rrif_spend=liquidation_gross,
                    tax_total=tax_paid,
                    net_total=liquidation_gross - tax_paid,
                )

                self._record_row(
                    year=year,
                    age_y=age_y,
                    age_s=age_s,
                    inc=inc,
                    goal=0,
                    bracket_ceiling=0,
                    bracket_room_household=0,
                    mandatory_min=0,
                    pots_start=self.bal,
                    plan=terminal_plan,
                    bal_end=Balances(0, 0, 0),
                    alive_y=False,
                    alive_s=False,
                    estate_tax=tax_paid,
                    net_estate=net_estate,
                    peak_future_rate=0.0,
                )
                break

            self._update_tfsa_room_for_new_year(i)

            balances_start = Balances(self.bal.rrif, self.bal.tfsa, self.bal.nonreg)
            pots = self._pots_for_year()

            inc = self._fixed_incomes_for_year(age_y, age_s, alive_y, alive_s)
            goal = self._goal_for_year_index(i)

            self.logger.info(
                f"Fixed Income: ${inc.fy_cpp + inc.fy_oas + inc.fy_pen + inc.fs_cpp + inc.fs_oas + inc.fs_pen:,.2f} "
                f"| Goal: ${goal:,.2f}"
            )

            bracket_ceiling, bracket_room_household, peak_future_rate = self._bracket_room_household(inc, i, alive_y, alive_s)
            mandatory_min = self._rrif_mandatory_min(age_y, pots.rrif_pot)

            plan = self._plan_withdrawals_nonreg_then_rrif_then_tfsa_then_bracket_fill(
                age_y=age_y,
                age_s=age_s,
                inc=inc,
                pots=pots,
                goal=goal,
                bracket_room_household=bracket_room_household,
                mandatory_min=mandatory_min,
                peak_future_rate=peak_future_rate,
                alive_y=alive_y,
                alive_s=alive_s,
            )

            metric = self._metric_cash(plan.gross_total, plan.net_total)
            if self.run_out_age_you is None and metric + 1e-6 < goal:
                self.run_out_age_you = age_y

            new_bal, next_recontrib = self._apply_plan_to_balances(pots, plan)
            self.bal = new_bal
            self.next_year_recontribution = next_recontrib

            self._record_row(
                year=year,
                age_y=age_y,
                age_s=age_s,
                inc=inc,
                goal=goal,
                bracket_ceiling=bracket_ceiling,
                bracket_room_household=bracket_room_household,
                mandatory_min=mandatory_min,
                pots_start=balances_start,
                plan=plan,
                bal_end=self.bal,
                alive_y=alive_y,
                alive_s=alive_s,
                peak_future_rate=peak_future_rate,
            )

        return self.rows, self.run_out_age_you


def run_model_simulation(
    settings_input: Dict, progress_cb: Optional[Callable[[float, str], None]] = None
) -> Tuple[List[Dict], Optional[int]]:
    """
    Backwards-compatible entry point.

    New knobs you can pass in settings_input:
      - "nonreg_real_return_pct": <number>  # e.g. 2.5 for 2.5% real
      - "dynamic_melt_safety_margin": <number>  # e.g. 0.08 for 8 percentage points

    NEW (dynamic mode only):
      - "dynamic_global_opt_enabled": True/False  # default True
      - "dynamic_global_opt_estate_weight": float # default 1.0
      - "dynamic_global_opt_target_r_grid": list[float]  # optional
      - "dynamic_global_opt_safety_grid": list[float]    # optional
    """
    sim = RetirementSimulator(settings_input, progress_cb=progress_cb)
    return sim.run()