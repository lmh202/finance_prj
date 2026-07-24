# FNSPID + FinBERT news risk-overlay research

The overlay predicts the five-day realized/HAR variance ratio with a
Gamma-log model, whose deviance matches QLIKE. Feature family,
regularization strength, and amplification
constraint are selected on 2019-2020 validation only.

## Result

- Selected family: `legacy+source+events`
- Mode: `two_sided`
- Validation QLIKE gain: +7.46%
- Validation gain in 2019: +4.63%
- Validation gain in 2020: +8.32%
- Worst validation-year gain: +4.63%
- Validation decision: **PASS** (minimum gain 1.0%)
- 2021-2023 confirmation gain: +6.04%
- Confirmation DM: t=-3.00, p=0.00271

The confirmation period is not described as pristine because earlier
project experiments had already inspected 2021-2023.

## Validation screen

| Family | Mode | Alpha | 2019 gain | 2020 gain | Worst-year | Aggregate gain | DM p |
|---|---|---:|---:|---:|---:|---:|---:|
| attention+sentiment | two_sided | 0.1 | +4.22% | +33.41% | +4.22% | +26.66% | 0.138 |
| attention+sentiment | amplify_only | 0.1 | +4.19% | +33.41% | +4.19% | +26.65% | 0.138 |
| attention | two_sided | 0.1 | +4.20% | +33.33% | +4.20% | +26.59% | 0.138 |
| attention | amplify_only | 0.1 | +4.19% | +33.33% | +4.19% | +26.59% | 0.138 |
| attention+events | two_sided | 0.1 | +4.35% | +32.85% | +4.35% | +26.26% | 0.143 |
| attention+events | amplify_only | 0.1 | +4.32% | +32.86% | +4.32% | +26.26% | 0.143 |
| expanded_all | amplify_only | 10 | +4.50% | +12.71% | +4.50% | +10.81% | 0.0158 |
| expanded_all | two_sided | 10 | +4.50% | +12.71% | +4.50% | +10.81% | 0.0158 |
| attention+source | amplify_only | 10 | +4.38% | +12.47% | +4.38% | +10.59% | 0.0156 |
| attention+source | two_sided | 10 | +4.38% | +12.47% | +4.38% | +10.59% | 0.0156 |
| legacy+expanded | amplify_only | 10 | +4.63% | +8.51% | +4.63% | +7.61% | 0.0297 |
| legacy+expanded | two_sided | 10 | +4.63% | +8.51% | +4.63% | +7.61% | 0.0297 |
| legacy+sentiment | two_sided | 1 | +4.37% | +8.52% | +4.37% | +7.56% | 0.0277 |
| legacy+sentiment | amplify_only | 1 | +4.37% | +8.52% | +4.37% | +7.56% | 0.0277 |
| legacy+source+events | amplify_only | 10 | +4.63% | +8.32% | +4.63% | +7.46% | 0.0281 |
| legacy+source+events | two_sided | 10 | +4.63% | +8.32% | +4.63% | +7.46% | 0.0281 |
| legacy+source | amplify_only | 10 | +4.58% | +8.33% | +4.58% | +7.46% | 0.0285 |
| legacy+source | two_sided | 10 | +4.58% | +8.33% | +4.58% | +7.46% | 0.0285 |
| legacy | two_sided | 1 | +4.45% | +8.22% | +4.45% | +7.35% | 0.0241 |
| legacy | amplify_only | 1 | +4.45% | +8.22% | +4.45% | +7.35% | 0.0241 |
| legacy+events | amplify_only | 1 | +4.51% | +8.06% | +4.51% | +7.24% | 0.0241 |
| legacy+events | two_sided | 1 | +4.51% | +8.06% | +4.51% | +7.24% | 0.0241 |

## Largest selected coefficients

| Feature | Scaled coefficient | Raw coefficient |
|---|---:|---:|
| has_news | +0.0164 | +0.0341 |
| title_token_novelty | +0.0147 | +0.0382 |
| firm_specific_share | +0.0143 | +0.0304 |
| mean_story_breadth | +0.0142 | +0.0227 |
| max_story_breadth | +0.0124 | +0.0152 |
| log_count | +0.0121 | +0.0162 |
| event_earnings_share | +0.0110 | +0.0384 |
| unique_publisher_count | +0.0095 | +0.0069 |
| mean_text_length | +0.0093 | +0.0000 |
| publisher_missing_share | +0.0083 | +0.0191 |
| summary_share | +0.0083 | +0.0191 |
| sent_std | +0.0077 | +0.0269 |
| publisher_entropy | +0.0067 | +0.0151 |
| unique_story_count | +0.0051 | +0.0017 |
| event_corporate_action_share | +0.0039 | +0.0186 |

## Guardrails

- FNSPID dates are shifted strictly to the next trading session.
- Only symbol-dates inside actual FNSPID coverage are evaluated.
- Intraday timing is excluded because >99% of timestamps are midnight.
- The candidate does not overwrite the deployed HAR checkpoint.