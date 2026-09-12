# Regression Tests

Every bug fix — including issues introduced and caught mid-refactor — must add a dedicated regression test file named `test_<symptom>_regression.py`.

## Mandatory Requirements:

1. **Fail on Pre-fix Code:** Prove the test fails before applying the fix (revert, verify RED, restore, verify GREEN). A regression test never seen failing is an assumption, not a guard.
2. **Documented Context:** The module/function docblock must explicitly document:
   - What broke
   - Why it mattered
   - Defect / ticket identifier or context
3. **Counterweight Assertion:** Must include a counterweight case asserting the fix did not overshoot or break complementary behaviors (e.g. "fixing delete move on boundary cases does not break single-facility deletion prohibition").
