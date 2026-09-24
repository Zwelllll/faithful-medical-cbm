# Stage 11: intervention policies and development curves

Completed 2026-09-24. All policies evaluated on the same 658 development cases using the frozen fold-specific heads. No CNN/LR retraining, concept inference, locked-test images/targets, intervention-aware fitting or joint CBM work.

## Frozen policy definitions

- **Random-error ORACLE / NON-DEPLOYABLE:** identify errors using truth, then uniformly permute remaining errors without replacement. Seed 42; 100 independently seeded repetitions. PCG64 with SeedSequence([42, repetition]); process cases in frozen development ID order. Use paired random orders across soft/hard, with independent streams across repetitions.
- **Confidently-wrong ORACLE / NON-DEPLOYABLE:** query remaining errors by descending abs(original q - 0.5), ties resolved by frozen concept order.
- **Active:** select the unqueried concept maximizing (1 - 2*abs(original q - 0.5)) * abs(p_force_1_current - p_force_0_current). Recompute forced-value effects against the current cumulative state and the appropriate soft/hard fold head. Ties use frozen concept order. Reveal only the selected target afterward.

The user explicitly froze this active formula for Stage 11. It supersedes PROJECT_SPEC section 15's earlier entropy/expected-impact formula for this experiment; decision D016 records the change without rewriting the historical specification.

Oracle policies stop when errors are exhausted and pad later budgets with unchanged predictions and blank selection/reveal fields. Active makes seven queries without repeats, including threshold-correct concepts. Hard uses binary inputs; soft leaves unqueried probabilities untouched. All diagnosis thresholds remain >=0.5. The active selector receives only immutable original probabilities, remaining names and current forced-value probabilities, never ground truth, diagnosis labels or an executor reference.

Oracle error knowledge is an upper-information reference, not a mathematical guarantee of optimal diagnosis performance. Corrections are assumed perfectly accurate; no clinical utility or causal claims are made.

## Metrics and denominators

At each k=0..7, compute pooled metrics over 658 cases using the frozen Stage 8 definitions, including positive-probability 10-bin ECE. Raw trajectory/selection-order outputs precede aggregate calculation. Random policy summaries are mean +/- sample SD (ddof=1) of 100 per-repetition metrics, not metrics of averaged predictions and not patient-level confidence intervals. Deterministic policies have SD zero.

- Query precision: cumulative actual errors corrected / actual queries; undefined at k=0.
- Mean corrected errors: cumulative error count / 658.
- Diagnosis correction/improvement rate: originally wrong diagnoses now correct / originally wrong diagnoses (soft denominator 118; hard 133).
- Harm rate: originally correct diagnoses now wrong / originally correct diagnoses (soft denominator 540; hard 525).
- Change rate: fraction of all 658 cases with diagnosis different from k=0.

## Integrity and numerical reproduction

All budgets/repetitions have exact development coverage; zero test-ID overlap. Saved heads verify validation exclusion from LR training, exact feature order and frozen OOF source. Full-development heads are never used. All Stage 7-10 inputs/output hashes remain unchanged. k=0 probability errors are at most 3.33e-16 (soft), 2.22e-16 (hard), below 1e-12. k=0 metric errors are below 1e-12. No selected concept repeats.

## Curves

### Soft diagnosis metrics

| Policy | k | AUROC | Macro-F1 | Accuracy | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random-error ORACLE | 0 | 0.8635 +/- 0.0000 | 0.7720 +/- 0.0000 | 0.8207 +/- 0.0000 | 0.5960 +/- 0.0000 | 0.9174 +/- 0.0000 | 0.1279 +/- 0.0000 | 0.0249 +/- 0.0000 |
| Random-error ORACLE | 1 | 0.8733 +/- 0.0061 | 0.7617 +/- 0.0082 | 0.8019 +/- 0.0068 | 0.6503 +/- 0.0158 | 0.8671 +/- 0.0076 | 0.1292 +/- 0.0028 | 0.0563 +/- 0.0058 |
| Random-error ORACLE | 2 | 0.8808 +/- 0.0051 | 0.7471 +/- 0.0079 | 0.7786 +/- 0.0069 | 0.7072 +/- 0.0155 | 0.8093 +/- 0.0068 | 0.1325 +/- 0.0027 | 0.0809 +/- 0.0059 |
| Random-error ORACLE | 3 | 0.8868 +/- 0.0029 | 0.7578 +/- 0.0039 | 0.7845 +/- 0.0036 | 0.7519 +/- 0.0074 | 0.7985 +/- 0.0043 | 0.1317 +/- 0.0016 | 0.0881 +/- 0.0037 |
| Random-error ORACLE | 4 | 0.8886 +/- 0.0014 | 0.7575 +/- 0.0017 | 0.7831 +/- 0.0017 | 0.7616 +/- 0.0030 | 0.7923 +/- 0.0021 | 0.1313 +/- 0.0008 | 0.0876 +/- 0.0018 |
| Random-error ORACLE | 5 | 0.8873 +/- 0.0008 | 0.7538 +/- 0.0000 | 0.7796 +/- 0.0000 | 0.7576 +/- 0.0000 | 0.7891 +/- 0.0000 | 0.1321 +/- 0.0004 | 0.0906 +/- 0.0010 |
| Random-error ORACLE | 6 | 0.8867 +/- 0.0003 | 0.7538 +/- 0.0000 | 0.7796 +/- 0.0000 | 0.7576 +/- 0.0000 | 0.7891 +/- 0.0000 | 0.1325 +/- 0.0001 | 0.0901 +/- 0.0002 |
| Random-error ORACLE | 7 | 0.8865 +/- 0.0000 | 0.7538 +/- 0.0000 | 0.7796 +/- 0.0000 | 0.7576 +/- 0.0000 | 0.7891 +/- 0.0000 | 0.1326 +/- 0.0000 | 0.0903 +/- 0.0000 |
| Confidently-wrong ORACLE | 0 | 0.8635 | 0.7720 | 0.8207 | 0.5960 | 0.9174 | 0.1279 | 0.0249 |
| Confidently-wrong ORACLE | 1 | 0.8619 | 0.7522 | 0.7903 | 0.6616 | 0.8457 | 0.1359 | 0.0575 |
| Confidently-wrong ORACLE | 2 | 0.8779 | 0.7390 | 0.7720 | 0.6919 | 0.8065 | 0.1344 | 0.0835 |
| Confidently-wrong ORACLE | 3 | 0.8827 | 0.7547 | 0.7812 | 0.7525 | 0.7935 | 0.1340 | 0.0827 |
| Confidently-wrong ORACLE | 4 | 0.8894 | 0.7572 | 0.7827 | 0.7626 | 0.7913 | 0.1310 | 0.0883 |
| Confidently-wrong ORACLE | 5 | 0.8877 | 0.7538 | 0.7796 | 0.7576 | 0.7891 | 0.1320 | 0.0920 |
| Confidently-wrong ORACLE | 6 | 0.8865 | 0.7538 | 0.7796 | 0.7576 | 0.7891 | 0.1326 | 0.0903 |
| Confidently-wrong ORACLE | 7 | 0.8865 | 0.7538 | 0.7796 | 0.7576 | 0.7891 | 0.1326 | 0.0903 |
| Active | 0 | 0.8635 | 0.7720 | 0.8207 | 0.5960 | 0.9174 | 0.1279 | 0.0249 |
| Active | 1 | 0.8840 | 0.7836 | 0.8161 | 0.7121 | 0.8609 | 0.1250 | 0.0370 |
| Active | 2 | 0.8910 | 0.7771 | 0.8070 | 0.7323 | 0.8391 | 0.1239 | 0.0627 |
| Active | 3 | 0.9005 | 0.7783 | 0.8100 | 0.7172 | 0.8500 | 0.1184 | 0.0566 |
| Active | 4 | 0.9055 | 0.7940 | 0.8237 | 0.7374 | 0.8609 | 0.1156 | 0.0518 |
| Active | 5 | 0.9061 | 0.8001 | 0.8283 | 0.7525 | 0.8609 | 0.1150 | 0.0495 |
| Active | 6 | 0.9057 | 0.8001 | 0.8283 | 0.7525 | 0.8609 | 0.1158 | 0.0509 |
| Active | 7 | 0.9046 | 0.7981 | 0.8267 | 0.7475 | 0.8609 | 0.1166 | 0.0557 |

### Soft query and diagnosis effects

| Policy | k | Query precision | Mean queries | Mean errors corrected | Diagnosis change | Diagnosis improvement | Diagnosis harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random-error ORACLE | 0 | undefined | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| Random-error ORACLE | 1 | 1.0000 +/- 0.0000 | 0.7112 +/- 0.0000 | 0.7112 +/- 0.0000 | 0.1298 +/- 0.0084 | 0.3095 +/- 0.0305 | 0.0905 +/- 0.0065 |
| Random-error ORACLE | 2 | 1.0000 +/- 0.0000 | 1.1444 +/- 0.0000 | 1.1444 +/- 0.0000 | 0.2184 +/- 0.0068 | 0.4915 +/- 0.0244 | 0.1587 +/- 0.0064 |
| Random-error ORACLE | 3 | 1.0000 +/- 0.0000 | 1.3632 +/- 0.0000 | 1.3632 +/- 0.0000 | 0.2545 +/- 0.0038 | 0.6086 +/- 0.0118 | 0.1771 +/- 0.0037 |
| Random-error ORACLE | 4 | 1.0000 +/- 0.0000 | 1.4438 +/- 0.0000 | 1.4438 +/- 0.0000 | 0.2655 +/- 0.0017 | 0.6356 +/- 0.0000 | 0.1847 +/- 0.0020 |
| Random-error ORACLE | 5 | 1.0000 +/- 0.0000 | 1.4635 +/- 0.0000 | 1.4635 +/- 0.0000 | 0.2690 +/- 0.0000 | 0.6356 +/- 0.0000 | 0.1889 +/- 0.0000 |
| Random-error ORACLE | 6 | 1.0000 +/- 0.0000 | 1.4681 +/- 0.0000 | 1.4681 +/- 0.0000 | 0.2690 +/- 0.0000 | 0.6356 +/- 0.0000 | 0.1889 +/- 0.0000 |
| Random-error ORACLE | 7 | 1.0000 +/- 0.0000 | 1.4696 +/- 0.0000 | 1.4696 +/- 0.0000 | 0.2690 +/- 0.0000 | 0.6356 +/- 0.0000 | 0.1889 +/- 0.0000 |
| Confidently-wrong ORACLE | 0 | undefined | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Confidently-wrong ORACLE | 1 | 1.0000 | 0.7112 | 0.7112 | 0.1520 | 0.3390 | 0.1111 |
| Confidently-wrong ORACLE | 2 | 1.0000 | 1.1444 | 1.1444 | 0.2340 | 0.5169 | 0.1722 |
| Confidently-wrong ORACLE | 3 | 1.0000 | 1.3632 | 1.3632 | 0.2614 | 0.6186 | 0.1833 |
| Confidently-wrong ORACLE | 4 | 1.0000 | 1.4438 | 1.4438 | 0.2660 | 0.6356 | 0.1852 |
| Confidently-wrong ORACLE | 5 | 1.0000 | 1.4635 | 1.4635 | 0.2690 | 0.6356 | 0.1889 |
| Confidently-wrong ORACLE | 6 | 1.0000 | 1.4681 | 1.4681 | 0.2690 | 0.6356 | 0.1889 |
| Confidently-wrong ORACLE | 7 | 1.0000 | 1.4696 | 1.4696 | 0.2690 | 0.6356 | 0.1889 |
| Active | 0 | undefined | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Active | 1 | 0.3571 | 1.0000 | 0.3571 | 0.1535 | 0.4153 | 0.0963 |
| Active | 2 | 0.3313 | 2.0000 | 0.6626 | 0.1717 | 0.4407 | 0.1130 |
| Active | 3 | 0.3034 | 3.0000 | 0.9103 | 0.2112 | 0.5593 | 0.1352 |
| Active | 4 | 0.2698 | 4.0000 | 1.0790 | 0.1884 | 0.5339 | 0.1130 |
| Active | 5 | 0.2526 | 5.0000 | 1.2629 | 0.2021 | 0.5847 | 0.1185 |
| Active | 6 | 0.2345 | 6.0000 | 1.4073 | 0.2112 | 0.6102 | 0.1241 |
| Active | 7 | 0.2099 | 7.0000 | 1.4696 | 0.2097 | 0.6017 | 0.1241 |

### Hard diagnosis metrics

| Policy | k | AUROC | Macro-F1 | Accuracy | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random-error ORACLE | 0 | 0.8402 +/- 0.0000 | 0.7504 +/- 0.0000 | 0.7979 +/- 0.0000 | 0.6010 +/- 0.0000 | 0.8826 +/- 0.0000 | 0.1335 +/- 0.0000 | 0.0453 +/- 0.0000 |
| Random-error ORACLE | 1 | 0.8695 +/- 0.0058 | 0.7767 +/- 0.0080 | 0.8171 +/- 0.0065 | 0.6509 +/- 0.0151 | 0.8887 +/- 0.0073 | 0.1287 +/- 0.0025 | 0.0589 +/- 0.0078 |
| Random-error ORACLE | 2 | 0.8890 +/- 0.0048 | 0.7692 +/- 0.0074 | 0.8055 +/- 0.0062 | 0.6798 +/- 0.0130 | 0.8595 +/- 0.0059 | 0.1255 +/- 0.0024 | 0.0648 +/- 0.0052 |
| Random-error ORACLE | 3 | 0.9011 +/- 0.0031 | 0.7827 +/- 0.0053 | 0.8145 +/- 0.0045 | 0.7175 +/- 0.0099 | 0.8562 +/- 0.0045 | 0.1206 +/- 0.0017 | 0.0834 +/- 0.0042 |
| Random-error ORACLE | 4 | 0.9051 +/- 0.0015 | 0.7932 +/- 0.0020 | 0.8222 +/- 0.0019 | 0.7431 +/- 0.0036 | 0.8563 +/- 0.0024 | 0.1186 +/- 0.0009 | 0.0827 +/- 0.0019 |
| Random-error ORACLE | 5 | 0.9042 +/- 0.0008 | 0.7917 +/- 0.0011 | 0.8205 +/- 0.0009 | 0.7459 +/- 0.0023 | 0.8525 +/- 0.0008 | 0.1191 +/- 0.0005 | 0.0875 +/- 0.0015 |
| Random-error ORACLE | 6 | 0.9032 +/- 0.0004 | 0.7900 +/- 0.0000 | 0.8191 +/- 0.0000 | 0.7424 +/- 0.0000 | 0.8522 +/- 0.0000 | 0.1197 +/- 0.0003 | 0.0880 +/- 0.0012 |
| Random-error ORACLE | 7 | 0.9028 +/- 0.0000 | 0.7900 +/- 0.0000 | 0.8191 +/- 0.0000 | 0.7424 +/- 0.0000 | 0.8522 +/- 0.0000 | 0.1199 +/- 0.0000 | 0.0870 +/- 0.0000 |
| Confidently-wrong ORACLE | 0 | 0.8402 | 0.7504 | 0.7979 | 0.6010 | 0.8826 | 0.1335 | 0.0453 |
| Confidently-wrong ORACLE | 1 | 0.8559 | 0.7625 | 0.8040 | 0.6414 | 0.8739 | 0.1366 | 0.0661 |
| Confidently-wrong ORACLE | 2 | 0.8820 | 0.7651 | 0.8009 | 0.6818 | 0.8522 | 0.1295 | 0.0599 |
| Confidently-wrong ORACLE | 3 | 0.8962 | 0.7765 | 0.8100 | 0.7020 | 0.8565 | 0.1230 | 0.0776 |
| Confidently-wrong ORACLE | 4 | 0.9053 | 0.7930 | 0.8222 | 0.7424 | 0.8565 | 0.1184 | 0.0830 |
| Confidently-wrong ORACLE | 5 | 0.9047 | 0.7921 | 0.8207 | 0.7475 | 0.8522 | 0.1188 | 0.0887 |
| Confidently-wrong ORACLE | 6 | 0.9031 | 0.7900 | 0.8191 | 0.7424 | 0.8522 | 0.1197 | 0.0895 |
| Confidently-wrong ORACLE | 7 | 0.9028 | 0.7900 | 0.8191 | 0.7424 | 0.8522 | 0.1199 | 0.0870 |
| Active | 0 | 0.8402 | 0.7504 | 0.7979 | 0.6010 | 0.8826 | 0.1335 | 0.0453 |
| Active | 1 | 0.8699 | 0.7565 | 0.8009 | 0.6212 | 0.8783 | 0.1292 | 0.0569 |
| Active | 2 | 0.8889 | 0.7802 | 0.8191 | 0.6616 | 0.8870 | 0.1223 | 0.0532 |
| Active | 3 | 0.8988 | 0.7762 | 0.8131 | 0.6768 | 0.8717 | 0.1195 | 0.0736 |
| Active | 4 | 0.9028 | 0.7815 | 0.8146 | 0.7071 | 0.8609 | 0.1192 | 0.0771 |
| Active | 5 | 0.9037 | 0.7827 | 0.8146 | 0.7172 | 0.8565 | 0.1190 | 0.0831 |
| Active | 6 | 0.9028 | 0.7880 | 0.8176 | 0.7374 | 0.8522 | 0.1199 | 0.0851 |
| Active | 7 | 0.9028 | 0.7900 | 0.8191 | 0.7424 | 0.8522 | 0.1199 | 0.0870 |

### Hard query and diagnosis effects

| Policy | k | Query precision | Mean queries | Mean errors corrected | Diagnosis change | Diagnosis improvement | Diagnosis harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random-error ORACLE | 0 | undefined | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| Random-error ORACLE | 1 | 1.0000 +/- 0.0000 | 0.7112 +/- 0.0000 | 0.7112 +/- 0.0000 | 0.1082 +/- 0.0074 | 0.3152 +/- 0.0246 | 0.0557 +/- 0.0061 |
| Random-error ORACLE | 2 | 1.0000 +/- 0.0000 | 1.1444 +/- 0.0000 | 1.1444 +/- 0.0000 | 0.1585 +/- 0.0060 | 0.4109 +/- 0.0204 | 0.0946 +/- 0.0056 |
| Random-error ORACLE | 3 | 1.0000 +/- 0.0000 | 1.3632 +/- 0.0000 | 1.3632 +/- 0.0000 | 0.1962 +/- 0.0043 | 0.5265 +/- 0.0153 | 0.1126 +/- 0.0039 |
| Random-error ORACLE | 4 | 1.0000 +/- 0.0000 | 1.4438 +/- 0.0000 | 1.4438 +/- 0.0000 | 0.2108 +/- 0.0019 | 0.5817 +/- 0.0063 | 0.1168 +/- 0.0017 |
| Random-error ORACLE | 5 | 1.0000 +/- 0.0000 | 1.4635 +/- 0.0000 | 1.4635 +/- 0.0000 | 0.2145 +/- 0.0009 | 0.5865 +/- 0.0000 | 0.1203 +/- 0.0012 |
| Random-error ORACLE | 6 | 1.0000 +/- 0.0000 | 1.4681 +/- 0.0000 | 1.4681 +/- 0.0000 | 0.2158 +/- 0.0000 | 0.5865 +/- 0.0000 | 0.1219 +/- 0.0000 |
| Random-error ORACLE | 7 | 1.0000 +/- 0.0000 | 1.4696 +/- 0.0000 | 1.4696 +/- 0.0000 | 0.2158 +/- 0.0000 | 0.5865 +/- 0.0000 | 0.1219 +/- 0.0000 |
| Confidently-wrong ORACLE | 0 | undefined | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Confidently-wrong ORACLE | 1 | 1.0000 | 0.7112 | 0.7112 | 0.1185 | 0.3083 | 0.0705 |
| Confidently-wrong ORACLE | 2 | 1.0000 | 1.1444 | 1.1444 | 0.1641 | 0.4135 | 0.1010 |
| Confidently-wrong ORACLE | 3 | 1.0000 | 1.3632 | 1.3632 | 0.1915 | 0.5038 | 0.1124 |
| Confidently-wrong ORACLE | 4 | 1.0000 | 1.4438 | 1.4438 | 0.2067 | 0.5714 | 0.1143 |
| Confidently-wrong ORACLE | 5 | 1.0000 | 1.4635 | 1.4635 | 0.2143 | 0.5865 | 0.1200 |
| Confidently-wrong ORACLE | 6 | 1.0000 | 1.4681 | 1.4681 | 0.2158 | 0.5865 | 0.1219 |
| Confidently-wrong ORACLE | 7 | 1.0000 | 1.4696 | 1.4696 | 0.2158 | 0.5865 | 0.1219 |
| Active | 0 | undefined | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Active | 1 | 0.3541 | 1.0000 | 0.3541 | 0.0881 | 0.2256 | 0.0533 |
| Active | 2 | 0.3283 | 2.0000 | 0.6565 | 0.1277 | 0.3684 | 0.0667 |
| Active | 3 | 0.3055 | 3.0000 | 0.9164 | 0.1520 | 0.4135 | 0.0857 |
| Active | 4 | 0.2815 | 4.0000 | 1.1261 | 0.1930 | 0.5188 | 0.1105 |
| Active | 5 | 0.2584 | 5.0000 | 1.2918 | 0.2021 | 0.5414 | 0.1162 |
| Active | 6 | 0.2333 | 6.0000 | 1.3997 | 0.2112 | 0.5714 | 0.1200 |
| Active | 7 | 0.2099 | 7.0000 | 1.4696 | 0.2158 | 0.5865 | 0.1219 |

## Descriptive area under intervention curves

Unnormalized trapezoidal area across budgets 0..7 (unit budget spacing; maximum possible area 7). Calculate each repetition's area first, then summarize mean/SD. These are descriptive curve summaries, not causal effects or deployment utility scores.

| Model | Policy | Area: AUROC curve | Area: Macro-F1 curve | Area: accuracy curve |
|---|---|---:|---:|---:|
| soft | Random-error ORACLE | 6.1786 +/- 0.0106 | 5.2947 +/- 0.0124 | 5.5075 +/- 0.0110 |
| soft | Confidently-wrong ORACLE | 6.1612 | 5.2738 | 5.4856 |
| soft | Active | 6.2769 | 5.5182 | 5.7371 |
| hard | Random-error ORACLE | 6.2437 +/- 0.0106 | 5.4737 +/- 0.0139 | 5.7074 +/- 0.0119 |
| hard | Confidently-wrong ORACLE | 6.2187 | 5.4493 | 5.6854 |
| hard | Active | 6.2384 | 5.4354 | 5.6884 |

## Factual endpoint observations

At k=7, both oracle policies share endpoints because correction order no longer matters. Hard active also reaches that identical all-true-concept input vector. Soft oracle endpoints differ from soft active: oracle policies stop after correcting errors, leaving threshold-correct probabilities unchanged; active queries all concepts and snaps them to binary truths. This differs from the Stage 9 oracle classifier because the diagnosis head remains the saved soft/hard head.

Soft active: AUROC 0.8635 -> 0.9046; accuracy 0.8207 -> 0.8267. Of 118 originally wrong diagnoses, 71 become correct; of 540 originally correct, 67 become incorrect. Soft oracle endpoints: AUROC 0.8865, accuracy 0.7796; 75 improve and 102 are harmed.

Hard endpoints: AUROC 0.8402 -> 0.9028; accuracy 0.7979 -> 0.8191. Of 133 originally wrong diagnoses, 78 improve; of 525 originally correct, 64 are harmed.

Active cumulative query precision at k=1 is 35.71% soft / 35.41% hard; at k=7 both are 967/4606 = 20.99%. Oracle query precision is 100% whenever queries occur, by definition of their truth-aware eligibility rule. Both correct all 967 original concept errors by k=7, averaging 1.4696 per case. No winner/ranking declaration is made.

## Artifacts and compact random replay

All results live in artifacts/interventions/policies/:

- soft_random_error_orders.csv and hard_random_error_orders.csv: 65,800 rows each, one case/repetition order, representing 526,400 trajectory steps per model. Stored concept indices use the frozen order, zero-based; [] means no errors.
- soft_confidently_wrong_trajectories.csv, hard_confidently_wrong_trajectories.csv, soft_active_trajectories.csv, hard_active_trajectories.csv: 5,264 rows each.
- budget_metrics.csv: 1,632 model/policy/repetition/budget rows, with seven diagnosis metrics, query/error counts, improvement/harm counts and rates.
- intervention_curves.csv: 48 aggregated model/policy/budget rows, with mean and SD columns.
- auc_summary.json: per-repetition and mean/SD areas.
- summary.json: protocol, definitions, endpoints, limitations, RNG scheme and counts.
- integrity.json: source/config/output hashes, cross-fitting checks and provenance.

Trajectory schema: case_num, validation_fold, diagnosis_binary, model_type, policy, oracle_non_deployable, repetition, query_budget, selected_concept, selected_probability (original OOF q), revealed_ground_truth, selected_was_wrong, concepts_queried, errors_corrected, diagnosis_score, melanoma_probability, predicted_diagnosis, diagnosis_changed, diagnosis_correct, original_diagnosis_correct, uncertainty, downstream_impact, active_score. Blank selected/score fields denote k=0, oracle padding, or non-active diagnostics.

Replay any compact random row using frozen inputs without RNG:

```python
import json
from pathlib import Path
from faithful_medical_cbm.interventions.engine import InterventionEngine, CONCEPT_ORDER
from faithful_medical_cbm.interventions.policies import trajectory
engine = InterventionEngine(Path("configs/intervention_engine.json"))
# row is a csv.DictReader record from *_random_error_orders.csv
order = [CONCEPT_ORDER[j] for j in json.loads(row["concept_indices_in_query_order"])]
steps = trajectory(engine, row["case_num"], row["model_type"],
                   "random_error_oracle", int(row["repetition"]), replay_order=order)
```

The frozen heads/OOF hashes and exact query order fully determine every required trajectory field and metric. Replay validates that the order contains every known error exactly once.

## Implementation and tests

Created configs/intervention_policies.json, interventions/policies.py and interventions/run_policies.py under src/faithful_medical_cbm/, tests/test_intervention_policies.py, this report and the listed artifacts. Updated README, DECISIONS and .gitignore. Existing Stage 10 code/artifacts and PROJECT_SPEC remain unchanged.

Commands from repository root:

```powershell
$env:PYTHONPATH = 'src'
python -m faithful_medical_cbm.interventions.run_policies --config configs/intervention_policies.json
python -m unittest tests.test_intervention_policies tests.test_intervention_engine -v
```

Outputs refuse overwrite. 17 distinct tests pass (10 policy tests, seven engine tests), including saved-output verification: every active trajectory replayed; all random orders checked for error-only eligibility, uniqueness and 658-case coverage in each repetition; full seed/order replay for repetitions 0 and 99; saved curve/AUC aggregations recomputed from per-budget metrics. No CNN or LR fitting is used.

## Before further stages

No implementation blocker remains for Stage 11. These are development diagnostics with the existing non-nested CNN/LR evaluation and unverified patient-independence limitations. Simulated corrections assume perfect annotations; actual human querying may differ. Active formula changes, joint CBMs, intervention-aware training and locked-test evaluation require their own explicit next-stage authorization. No next-stage implementation was started.
