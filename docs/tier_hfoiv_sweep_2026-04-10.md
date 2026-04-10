# Risk Tier + HFOIV Sweep Results

**Date:** 2026-04-10
**Setup:** Standalone $80k/year, `--margin 1` (no margin constraint), walk-forward strategy files per year, `--risk-tiers`
**HFOIV config when on:** `--hfoiv --hfoiv-rolling 6 --hfoiv-lookback 90` (p70_x0.25_r6_lb90)

## Strategy Files (walk-forward, no look-ahead)

| Year | Strategy File |
|------|--------------|
| 2021 | wf-2020-test-2021-slotbl-non3.json |
| 2022 | wf-2020-2021-test-2022-slotbl-non3.json |
| 2023 | wf-2020-2022-test-2023-slotbl-non3.json |
| 2024 | wf-2020-2023-test-2024-slotbl-non3.json |
| 2025 | mixed-best-ev-wf-2020-2024-slotbl-non3.json |
| 2026 YTD | mixed-best-ev-wf-2020-2025-slotbl-non3.json (through 2026-04-07) |

## Tier Maps Tested

| Label | Small (5-10, 10-15) | Medium (15-20, 20-25, 25-30, 30-40) | Large (40-50, 50-200) |
|-------|--------------------|------------------------------------|----------------------|
| Baseline | 0.50% | 1.50% | 3.00% |
| Candidate A | 0.25% | 1.00% | 3.00% |
| Candidate B | 0.25% | 0.75% | 3.00% |
| Candidate C | 0.50% | 0.75% | 3.00% |
| Candidate D | 0.25% | 0.75% | 2.00% |

---

## Results: WITHOUT HFOIV

| Tier Map | 2021 P&L% | 2021 DD | 2022 P&L% | 2022 DD | 2023 P&L% | 2023 DD | 2024 P&L% | 2024 DD | 2025 P&L% | 2025 DD | 2026 P&L% | 2026 DD | Sum P&L% | Worst DD |
|----------|----------|---------|----------|---------|----------|---------|----------|---------|----------|---------|----------|---------|----------|----------|
| **Baseline 0.50/1.50/3.00** | +137.1% | 19.9% | +33.9% | 36.9% | +8.5% | 24.5% | -24.5% | 41.6% | +81.7% | 25.5% | +23.9% | 26.9% | +260.6% | 41.6% |
| **A: 0.25/1.00/3.00** | +63.8% | 12.4% | +31.0% | 22.8% | +7.5% | 19.0% | -4.5% | 29.8% | +79.1% | 18.9% | +14.2% | 23.3% | +191.1% | 29.8% |
| **B: 0.25/0.75/3.00** | +51.6% | 10.7% | +19.5% | 21.2% | +2.6% | 17.9% | -0.9% | 25.8% | +77.7% | 19.8% | +13.2% | 17.4% | +163.7% | 25.8% |
| **C: 0.50/0.75/3.00** | +61.6% | 11.9% | +17.1% | 24.6% | +7.0% | 19.9% | -2.6% | 29.1% | +72.4% | 18.8% | +12.5% | 18.1% | +168.0% | 29.1% |
| **D: 0.25/0.75/2.00** | +51.6% | 10.7% | +19.5% | 21.2% | +0.5% | 14.7% | +0.8% | 23.5% | +57.6% | 14.3% | +16.4% | 10.2% | +146.4% | 23.5% |

### Dollar P&L (noHFOIV, $80k start each year)

| Tier Map | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|----------|------|------|------|------|------|----------|
| **Baseline** | $109,643 | $27,151 | $6,839 | -$19,635 | $65,354 | $19,158 |
| **A: 0.25/1.00/3.00** | $51,012 | $24,816 | $5,988 | -$3,613 | $63,292 | $11,324 |
| **B: 0.25/0.75/3.00** | $41,289 | $15,613 | $2,110 | -$745 | $62,192 | $10,528 |
| **C: 0.50/0.75/3.00** | $49,285 | $13,715 | $5,618 | -$2,092 | $57,884 | $9,981 |
| **D: 0.25/0.75/2.00** | $41,289 | $15,613 | $382 | $620 | $46,096 | $13,152 |

---

## Results: WITH HFOIV

| Tier Map | 2021 P&L% | 2021 DD | 2022 P&L% | 2022 DD | 2023 P&L% | 2023 DD | 2024 P&L% | 2024 DD | 2025 P&L% | 2025 DD | 2026 P&L% | 2026 DD | Sum P&L% | Worst DD |
|----------|----------|---------|----------|---------|----------|---------|----------|---------|----------|---------|----------|---------|----------|----------|
| **Baseline 0.50/1.50/3.00** | +140.4% | 12.4% | +39.8% | 34.7% | +17.4% | 20.9% | -22.7% | 40.0% | +77.8% | 22.7% | +23.9% | 26.9% | +276.6% | 40.0% |
| **A: 0.25/1.00/3.00** | +63.7% | 10.9% | +29.8% | 22.4% | +5.7% | 18.0% | -6.3% | 29.8% | +59.8% | 18.8% | +14.2% | 23.3% | +166.9% | 29.8% |
| **B: 0.25/0.75/3.00** | +47.9% | 9.7% | +21.8% | 20.9% | +2.7% | 16.8% | -0.7% | 25.8% | +60.0% | 18.8% | +13.2% | 17.4% | +144.9% | 25.8% |
| **C: 0.50/0.75/3.00** | +59.7% | 11.0% | +25.2% | 23.2% | +5.1% | 19.9% | -4.1% | 29.8% | +61.7% | 18.2% | +12.5% | 18.1% | +160.1% | 29.8% |
| **D: 0.25/0.75/2.00** | +47.9% | 9.7% | +21.8% | 20.9% | +2.3% | 13.8% | +0.8% | 23.5% | +55.0% | 14.0% | +16.4% | 10.2% | +144.2% | 23.5% |

### Dollar P&L (HFOIV on, $80k start each year)

| Tier Map | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|----------|------|------|------|------|------|----------|
| **Baseline** | $112,284 | $31,855 | $13,958 | -$18,156 | $62,249 | $19,158 |
| **A: 0.25/1.00/3.00** | $50,996 | $23,876 | $4,592 | -$5,078 | $47,859 | $11,324 |
| **B: 0.25/0.75/3.00** | $38,335 | $17,479 | $2,196 | -$538 | $47,969 | $10,528 |
| **C: 0.50/0.75/3.00** | $47,751 | $20,126 | $4,116 | -$3,256 | $49,356 | $9,981 |
| **D: 0.25/0.75/2.00** | $38,335 | $17,479 | $1,821 | $620 | $43,970 | $13,152 |

---

## HFOIV Impact (delta = HFOIV on minus HFOIV off)

| Tier Map | 2021 dP&L% | 2021 dDD | 2022 dP&L% | 2022 dDD | 2023 dP&L% | 2023 dDD | 2024 dP&L% | 2024 dDD | 2025 dP&L% | 2025 dDD | 2026 dP&L% | 2026 dDD |
|----------|-----------|---------|-----------|---------|-----------|---------|-----------|---------|-----------|---------|-----------|---------|
| **Baseline** | +3.3% | -7.5% | +5.9% | -2.2% | +8.9% | -3.6% | +1.8% | -1.6% | -3.9% | -2.8% | 0.0% | 0.0% |
| **A** | -0.1% | -1.5% | -1.2% | -0.4% | -1.8% | -1.0% | -1.8% | 0.0% | -19.3% | -0.1% | 0.0% | 0.0% |
| **B** | -3.7% | -1.0% | +2.3% | -0.3% | +0.1% | -1.1% | +0.2% | 0.0% | -17.7% | -1.0% | 0.0% | 0.0% |
| **C** | -1.9% | -0.9% | +8.1% | -1.4% | -1.9% | 0.0% | -1.5% | +0.7% | -10.7% | -0.6% | 0.0% | 0.0% |
| **D** | -3.7% | -1.0% | +2.3% | -0.3% | +1.8% | -0.9% | 0.0% | 0.0% | -2.6% | -0.3% | 0.0% | 0.0% |

---

## Key Findings

### 1. All candidate tier maps hold up across all 6 years (no overfitting)
Every candidate improves 2024 dramatically and reduces DD in every year. The pattern is consistent, not a 2024-2025 artifact.

### 2. HFOIV helps baseline the most, hurts tighter maps in trending years
- At baseline tiers, HFOIV adds P&L in 4/6 years and reduces DD in 5/6.
- At tighter tiers, HFOIV disproportionately throttles 2025 P&L (up to -19pt for Candidate A) because it's shrinking already-smaller positions.
- DD reduction from HFOIV is modest (0-1.5pt) when tighter tiers are already doing the work.

### 3. Without HFOIV, Candidate A retains 97% of baseline 2025 P&L
`0.25/1.00/3.00` noHFOIV: +79.1% vs baseline +81.7% — only 2.6pt gap. With HFOIV the gap balloons to 18pt.

### 4. Candidate D is the only config where 2024 is positive
`0.25/0.75/2.00` turns 2024 from -24.5% to +0.8%, worst DD 23.5% (nearly half baseline's 41.6%). The safest profile by far.

### 5. The large bucket (3.0%) is the main P&L driver
Candidates B and D have identical small/medium tiers (0.25/0.75) but B has large=3.0% and D has large=2.0%. In 2025: B gets +77.7% vs D's +57.6%. The 1pt large-bucket difference is worth ~20pt of annual P&L.

---

## Recommendations

| Goal | Tier Map | HFOIV | Rationale |
|------|----------|-------|-----------|
| **Max P&L, accept high DD** | Baseline 0.50/1.50/3.00 | ON | Highest total return, but 40% worst DD |
| **Best risk-adjusted (noHFOIV)** | A: 0.25/1.00/3.00 | OFF | 97% of baseline 2025 P&L, worst DD 29.8% |
| **Best risk-adjusted (HFOIV)** | B: 0.25/0.75/3.00 | ON | Worst DD 25.8%, all years profitable except 2024 (-0.7%) |
| **Most conservative** | D: 0.25/0.75/2.00 | ON or OFF | Only config with positive 2024, worst DD 23.5%, 2026 DD 10.2% |

---

## Important Caveat

All runs used `--margin 1` (no margin constraint). Real trading with $33,000 NQ intraday margin on $80k will cap simultaneous contracts and likely reduce both P&L and DD. These results should be re-validated with `--margin 33000` before making live decisions.

---

## Raw Data

Full JSON results saved to `scripts/tier_hfoiv_sweep_results.json`.
Sweep script: `scripts/sweep_tier_hfoiv.py`.
