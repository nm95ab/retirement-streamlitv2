# Architecture & Coding Best Practices Review

I have performed a comprehensive review of the project's design and implementation. Overall, the codebase is **structurally sound and maintains high standards for logic accuracy**, but it is approaching a level of complexity where architectural refactoring would significantly improve maintainability.

---

## ✅ Strengths (Best Practices Followed)

### 1. Type Safety & Clarity
- **Dataclasses**: Excellent use of `@dataclass` for `Balances`, `Incomes`, and `WithdrawPlan`. This provides a schema-first approach to state and results, making the code much easier to reason about.
- **Type Hinting**: Consistent use of Python type hints (`List[Dict]`, `Optional[int]`) across the codebase minimizes runtime errors and improves IDE discoverability.

### 2. Functional Purity in the Tax Engine
- The core tax calculations in `retirement2_helpers.py` (e.g., `calc_fed_bpa`, `ontario_surtax`) are mostly side-effect-free (pure functions). This is a critical best practice in financial software as it makes individual components easy to unit test.

### 3. Separation of Concerns (High Level)
- There is a clear distinction between the **Tax Logic** (`retirement2_helpers.py`) and the **Simulation Logic** (`retirement_simulation.py`). This allows the tax engine to be updated for new laws without breaking the simulation timeline.

---

## ❌ Areas for Maturity (Recommended Refactors)

### 1. Strategy Pattern for Withdrawals
- **Observation**: The withdrawal sequence is currently hardcoded into method names (e.g., `_plan_withdrawals_nonreg_then_rrif_then_tfsa...`). 
- **Recommendation**: Implement a `WithdrawalStrategy` interface. This would allow the simulation to swap between "Spend Non-Reg First", "Melt RRIF First", or "Proportional Spending" without modifying the main simulation loop.

### 2. State Management
- **Observation**: The `RetirementSimulator` is highly stateful, tracking `self.bal` and `self.tfsa_room` across multiple methods. This makes it difficult to test a specific "Year 15" scenario in isolation.
- **Recommendation**: Transition towards a more functional "State Transition" model where the simulation loop looks like: `next_state = process_year(current_state, year_inputs)`.

### 3. Encapsulation (Rules vs. Data)
- **Observation**: Deeply nested functions in `project_future_peak_rate` and the use of string prefixes (e.g., `prefix_age_amount_max`) make the code brittle.
- **Recommendation**: Encapsulate tax rules into a `TaxPolicy` or `TaxRules` class instead of a flat dictionary. This would provide auto-completion for settings and internal validation logic.

### 4. Logging & Observability
- **Observation**: The simulation relies on returning large lists of dictionaries. If a calculation error occurs in Year 20, it is hard to trace the intermediate decisions.
- **Recommendation**: Integrate the standard Python `logging` module to capture the "rationale" behind specific withdrawal decisions in each step.

---

## 🛠 Refactor Roadmap (Proposed)

1. **Phase 1: Package Structure**: Move scripts into a `retirement/` package with clear `core/`, `tax/`, and `strategies/` sub-packages.
2. **Phase 2: Strategy Interface**: Abstract the withdrawal logic into interchangeable strategy classes.
3. **Phase 3: Validation Layer**: Add Pydantic or similar validation for user settings to prevent "Garbage In, Garbage Out" scenarios.

---
**Status**: Initial Review Complete (Feb 5, 2026).
