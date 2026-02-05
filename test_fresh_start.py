import unittest
import sys
import os
sys.path.append(os.getcwd())
from retirement_simulation import run_model_simulation, DEFAULTS

class TestFreshStart(unittest.TestCase):
    def test_rrif_melting(self):
        print("\n--- Testing RRIF Melting Behavior ---")
        settings = DEFAULTS.copy()
        settings["rrif_start_balance"] = 1_000_000.0 # High liability
        settings["goal_income"] = 60000.0 # Low spending goal
        
        # This setup creates a scenario where:
        # Fixed Income (CPP/OAS) covers most of spending.
        # But RRIF Death Tax (on $1M) implies ~50% tax later.
        # Current Tax (on $60k) is ~20%.
        # The Solver SHOULD withdraw extra RRIF (~$100k+) to smooth this out, 
        # generating a Surplus that goes to TFSA/NonReg.
        
        rows, _ = run_model_simulation(settings)
        
        r0 = rows[0]
        peak = r0["Peak Future Rate"]
        rrif_w = r0["RRIF Withdraw"]
        surplus_tfsa = r0["Surplus -> TFSA"]
        
        print(f"Year 0: Peak Future Rate = {peak:.2%}")
        print(f"Year 0: RRIF Withdraw = ${rrif_w:,.2f}")
        print(f"Year 0: Surplus to TFSA = ${surplus_tfsa:,.2f}")
        
        # Assertions
        self.assertGreater(peak, 0.40, "Peak rate should reflect high tax bracket (death).")
        self.assertGreater(rrif_w, 150_000.0, "Should aggressive melt RRIF.")
        self.assertGreater(surplus_tfsa, 0.0, "Should serve surplus to TFSA.")

if __name__ == "__main__":
    unittest.main()
