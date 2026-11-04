"""brctta -- risk-controlled test-time adaptation for underwater image enhancement.

Package layout
--------------
    data      stream definitions, the 256x256 whole-image protocol, manifests
    cache     materialised uint8 stream cache (memoisation of the protocol)
    model     frozen backbone + the small residual adapter that carries all plasticity
    losses    self-supervised objectives used to *propose* an update
    signals   evidence computed for one candidate update (risk features)
    risk      accept / keep / reset controller (calibrated on dev)
    adapt     the causal prequential loop, episodic mode, replay
    baselines frozen / always / random / oracle / simple-threshold / stochastic-restore
    metrics   PSNR / SSIM / MSE / UIQM / UCIQE, implemented here so they cannot drift
    evaluate  per-run summaries and the §9.2 negative-transfer metrics
"""

__version__ = "0.1.0"
