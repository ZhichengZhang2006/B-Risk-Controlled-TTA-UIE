# Acceptance report (§11)

**Verdict: `fail`** — 24 pass / 4 fail / 0 unverifiable (of 28).

Relative-shortfall tolerance: **5%**. Script SHA256: `d6c5e2d813fb9252…`


## Status fields

The four axes are reported separately and must not be collapsed into one score.

| field | verdict | detail |
| --- | --- | --- |
| performance | `fail` | 5/6 passed, 1 failed, 0 unverifiable |
| risk_control | `fail` | 9/12 passed, 3 failed, 0 unverifiable |
| reproducibility | `pass` | 6/6 passed, 0 failed, 0 unverifiable |
| novelty | `documented_not_machine_checkable` | docs/novelty_audit.md is an argument, not a measurement |

## Gates

| gate | result | detail |
| --- | --- | --- |
| `G_LSUI2UIEB_psnr` | FAIL | ours=20.9361; ref=22.0612; ref_arm=strong_frozen; rel_gap=0.0510; tolerance=0.0500; abs_gap=-1.1251; abs_gap_unit=dB; mse_per_image_ours={'n': 89, 'mean': 0.016661948580053685, 'median': 0.009385301575971377, 'p90': 0.03 |
| `G_LSUI2UIEB_ssim` | PASS | ours=0.8561; ref=0.8722; ref_arm=strong_frozen; rel_gap=0.0185; tolerance=0.0500; abs_gap=-0.0162; abs_gap_unit=SSIM (absolute); mse_per_image_ours={'n': 89, 'mean': 0.016661948580053685, 'median': 0.009385301575971377,  |
| `G_UIEB2LSUI_psnr` | PASS | ours=21.9886; ref=22.1777; ref_arm=source_frozen; rel_gap=0.0085; tolerance=0.0500; abs_gap=-0.1892; abs_gap_unit=dB; mse_per_image_ours={'n': 428, 'mean': 0.011583371757575546, 'median': 0.007507329696668739, 'p90': 0.0 |
| `G_UIEB2LSUI_ssim` | PASS | ours=0.8519; ref=0.8540; ref_arm=source_frozen; rel_gap=0.0025; tolerance=0.0500; abs_gap=-0.0021; abs_gap_unit=SSIM (absolute); mse_per_image_ours={'n': 428, 'mean': 0.011583371757575546, 'median': 0.007507329696668739, |
| `G_LSUI2UFO_psnr` | PASS | ours=27.6598; ref=28.1403; ref_arm=strong_frozen; rel_gap=0.0171; tolerance=0.0500; abs_gap=-0.4805; abs_gap_unit=dB; mse_per_image_ours={'n': 120, 'mean': 0.0020887036842255785, 'median': 0.0017244654634323076, 'p90': 0 |
| `G_LSUI2UFO_ssim` | PASS | ours=0.8650; ref=0.8753; ref_arm=strong_frozen; rel_gap=0.0118; tolerance=0.0500; abs_gap=-0.0104; abs_gap_unit=SSIM (absolute); mse_per_image_ours={'n': 120, 'mean': 0.0020887036842255785, 'median': 0.001724465463432307 |
| `G_LSUI2UIEB_replay` | PASS | {"max_abs_dpsnr": 0.0, "max_abs_dssim": 0.0, "action_histogram": {"keep": 89}} |
| `G_LSUI2UIEB_candidate_labels` | PASS | ok |
| `G_UIEB2LSUI_replay` | PASS | {"max_abs_dpsnr": 0.0, "max_abs_dssim": 0.0, "action_histogram": {"reset": 271, "keep": 77, "accept": 80}} |
| `G_UIEB2LSUI_candidate_labels` | PASS | ok |
| `G_LSUI2UFO_replay` | PASS | {"max_abs_dpsnr": 0.0, "max_abs_dssim": 0.0, "action_histogram": {"reset": 65, "accept": 27, "keep": 28}} |
| `G_LSUI2UFO_candidate_labels` | PASS | ok |
| `G_LSUI2UIEB_beats_always` | FAIL | active=no; note=method abstained: no committed updates, so the tail-risk claim is unsupported; ours_worst10=0.0000; always_worst10=-0.0778; ours_NAR0=0.0000; always_NAR0=0.3034; ours_worst10_cand=-0.0944; always_worst10_ |
| `G_LSUI2UIEB_positive_gain` | FAIL | ours_dPSNR_mean=0.0000; note=realised prequential gain must be strictly positive |
| `G_LSUI2UIEB_nondegenerate` | FAIL | accept_rate=0.0000 |
| `G_LSUI2UIEB_not_worse_than_frozen` | PASS | ours=20.9361; frozen=20.9361; frozen_arm=source_adapter |
| `G_UIEB2LSUI_beats_always` | PASS | active=yes; note=; ours_worst10=-0.0404; always_worst10=-0.0969; ours_NAR0=0.1145; always_NAR0=0.4626; ours_worst10_cand=-0.1089; always_worst10_cand=-0.0969 |
| `G_UIEB2LSUI_positive_gain` | PASS | ours_dPSNR_mean=0.0035; note=realised prequential gain must be strictly positive |
| `G_UIEB2LSUI_nondegenerate` | PASS | accept_rate=0.1869 |
| `G_UIEB2LSUI_not_worse_than_frozen` | PASS | ours=21.9886; frozen=21.9892; frozen_arm=source_adapter |
| `G_LSUI2UFO_beats_always` | PASS | active=yes; note=; ours_worst10=-0.0271; always_worst10=-0.0827; ours_NAR0=0.1917; always_NAR0=0.5917; ours_worst10_cand=-0.0602; always_worst10_cand=-0.0827 |
| `G_LSUI2UFO_positive_gain` | PASS | ours_dPSNR_mean=0.0008; note=realised prequential gain must be strictly positive |
| `G_LSUI2UFO_nondegenerate` | PASS | accept_rate=0.2250 |
| `G_LSUI2UFO_not_worse_than_frozen` | PASS | ours=27.6598; frozen=27.6623; frozen_arm=source_adapter |
| `G_ledger` | PASS | /home/user/BRCTTA/runs/experiment_ledger.csv |
| `G_freeze` | PASS | run scripts/12_freeze.py --check |
| `G_novelty` | PASS | docs/novelty_audit.md |
| `G_data_manifest` | PASS | runs/manifests/data_audit.json |

## Reference policy

- gate reference: strongest measured deployable arm under the frozen protocol (runs/comparison/comparison_set.json, oracle excluded)
- excluded from every gate: `oracle`, `target-supervised`, `published-other-protocol`

Target-supervised and published numbers, reported as coordinates only:

- **LSUI2UIEB**: {"target_supervision_reference": null, "published_same_protocol": null, "status": "reported as an absolute coordinate; NOT part of the §11 gate"}
- **UIEB2LSUI**: {"target_supervision_reference": null, "published_same_protocol": null, "status": "reported as an absolute coordinate; NOT part of the §11 gate"}
- **LSUI2UFO**: {"target_supervision_reference": null, "published_same_protocol": null, "status": "reported as an absolute coordinate; NOT part of the §11 gate"}

## Note

MSE is reported per image but never used as an acceptance criterion (spec §11). g_m = max(0, (S_m - R_m)/S_m) with S the strongest comparable arm; PSNR gaps are reported in dB. An unverifiable gate is never counted as passed.
