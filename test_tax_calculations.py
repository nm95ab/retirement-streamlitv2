import unittest
import json
from retirement2_helpers import (
    calc_total_tax,
    calc_fed_bpa,
    calc_age_amount,
    ontario_surtax,
    ontario_health_premium,
    tax_from_brackets
)

# ==========================================
# CONTROLLED TEST CONSTANTS
# Using simple base-10 numbers for predictability
# ==========================================
TEST_SETTINGS = {
    "fed_basic_personal_amount": 10000.0,
    "ont_basic_personal_amount": 5000.0,
    "fed_age_amount_max": 2000.0,
    "fed_age_amount_thresh": 40000.0,
    "ont_age_amount_max": 1000.0,
    "ont_age_amount_thresh": 40000.0,
    "fed_pension_amount_max": 2000.0,
    "ont_pension_amount_max": 1000.0,
    "ont_surtax_enabled": True,
    "ont_surtax_t1": 5000.0, # Basic Ont tax thresh
    "ont_surtax_t2": 10000.0,
    "ont_health_premium_enabled": True,
    "enable_oas_clawback": True,
    "oas_clawback_threshold": 90000.0,
}

# 10% Fed across the board
TEST_FED_BRACKETS = [[1000000, 0.10]] 
# 5% Ont across the board
TEST_ONT_BRACKETS = [[1000000, 0.05]]

class TestTaxCalculationsRobust(unittest.TestCase):
    def test_tax_from_brackets_logic(self):
        """Verifies progressive logic without specific 2026 rates."""
        simple_brackets = [[1000, 0.10], [2000, 0.20]]
        # 0 to 1000 @ 10% = 100
        # 1000 to 1500 @ 20% = 100
        # Total = 200
        self.assertEqual(tax_from_brackets(1500, simple_brackets), 200.0)

    def test_calc_fed_bpa_logic(self):
        """Verifies BPA phase-out logic using controlled thresholds."""
        # The function uses hardcoded thresholds (181440, 258482) for 2026.
        # We verify that it returns the expected max/min and does linear interpolation.
        settings = {"fed_basic_personal_amount": 20000.0}
        bpa_min = 14829.0 # Hardcoded in helper
        
        # Below phase-out
        self.assertEqual(calc_fed_bpa(100000, settings), 20000.0)
        # Above phase-out
        self.assertEqual(calc_fed_bpa(300000, settings), bpa_min)
        # Midpoint
        mid = (181440 + 258482) / 2
        self.assertAlmostEqual(calc_fed_bpa(mid, settings), (20000.0 + bpa_min) / 2)

    def test_calc_age_amount_logic(self):
        """Verifies Age Amount phase-out logic."""
        # $2000 max, $40,000 threshold, 15% reduction
        self.assertEqual(calc_age_amount(30000, 65, TEST_SETTINGS, "fed"), 2000.0)
        # $1000 over threshold = $150 reduction
        self.assertEqual(calc_age_amount(41000, 65, TEST_SETTINGS, "fed"), 2000.0 - 150.0)
        # Full reduction
        self.assertEqual(calc_age_amount(100000, 65, TEST_SETTINGS, "fed"), 0.0)

    def test_ontario_surtax_logic(self):
        """Verifies Surtax tiers logic."""
        # t1=5000 (20%), t2=10000 (36%)
        # Case 1: below t1
        self.assertEqual(ontario_surtax(4000, TEST_SETTINGS), 0.0)
        # Case 2: between t1 and t2 ($1000 over t1)
        self.assertEqual(ontario_surtax(6000, TEST_SETTINGS), 0.20 * 1000)
        # Case 3: above t2 ($6000 over t1, $1000 over t2)
        expected = 0.20 * 6000 + 0.36 * 1000 # 1200 + 360 = 1560
        self.assertEqual(ontario_surtax(11000, TEST_SETTINGS), expected)

    def test_calc_total_tax_integration_robust(self):
        """Verifies full tax engine using controlled rates."""
        income = 50000.0
        # Expected Logic:
        # Fed Tax Base: 50000 * 0.10 = 5000
        # Fed Credits: BPA(10000) * 0.10 = 1000
        # Net Fed = 4000
        #
        # Ont Tax Base: 50000 * 0.05 = 2500
        # Ont Credits: BPA(5000) * 0.05 = 250
        # Net Ont = 2250
        #
        # OHP: 50000 falls in [48k, 72k] -> min(600, 450 + 0.25*(50000-48000)) = 600
        # Surtax on 2250: below t1(5000) = 0
        # Total = 4000 + 2250 + 600 = 6850
        
        tax = calc_total_tax(income, 60, TEST_SETTINGS, TEST_FED_BRACKETS, TEST_ONT_BRACKETS)
        self.assertEqual(tax, 6850.0)

    def test_pension_credit_logic(self):
        """Verify pension credit uses lowest marginal rate as multiplier."""
        income = 50000.0
        # Pension credit = amount * lowest_rate
        # Fed: 2000 * 0.10 = 200
        # Ont: 1000 * 0.05 = 50
        # Total reduction = 250
        
        tax_no_pen = calc_total_tax(income, 65, TEST_SETTINGS, TEST_FED_BRACKETS, TEST_ONT_BRACKETS, 0)
        tax_pen = calc_total_tax(income, 65, TEST_SETTINGS, TEST_FED_BRACKETS, TEST_ONT_BRACKETS, 2000)
        
        self.assertAlmostEqual(tax_no_pen - tax_pen, 250.0)

    def test_oas_clawback_logic(self):
        """Verify OAS clawback is triggered correctly at 15%."""
        # Threshold 90000. Income 100000. diff = 10000. 15% = 1500.
        s = TEST_SETTINGS.copy()
        tax_under = calc_total_tax(90000, 65, s, TEST_FED_BRACKETS, TEST_ONT_BRACKETS)
        tax_over = calc_total_tax(100000, 65, s, TEST_FED_BRACKETS, TEST_ONT_BRACKETS)
        
        # Basic tax diff: (100000 - 90000) * (0.10 Fed + 0.05 Ont) = 1500
        # Clawback diff: 1500
        # Total diff: 3000
        self.assertAlmostEqual(tax_over - tax_under, 3000.0)

if __name__ == "__main__":
    unittest.main()
