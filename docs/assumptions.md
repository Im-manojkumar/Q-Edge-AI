# Modelling assumptions

Every number the cost model uses is listed here so the report's conclusions can be traced back to it and
changed in `configs/default.yaml`. Figures are order-of-magnitude values for a typical deployment, not
measurements of one specific machine. Prices and error rates change often; check the providers' current pages
before quoting them.

## Classical resources

| Resource | Throughput (parallel / serial GFLOPS) | Power | Memory | Network from edge | Price |
|---|---|---|---|---|---|
| Edge CPU (ARM board) | 25 / 8 | 6 W | 8 GB | local | 0 |
| Edge GPU (Jetson-class) | 600 / 4 | 15 W | 8 GB | local | 0 |
| Cloud CPU (32 vCPU VM) | 600 / 40 | 180 W | 128 GB | 40 ms RTT, 20 Mbps up | $1.50/h |
| Cloud GPU (A100-class) | 12 000 / 30 | 350 W | 80 GB | 40 ms RTT, 20 Mbps up | $3.00/h |

* "Serial" throughput is what work that does not parallelise (graph heuristics, small SVMs) gets.
* Edge radio: 2 W while transmitting. Edge energy = local compute energy + radio energy.

## Quantum resources

| Figure | Value | Basis |
|---|---|---|
| QPU qubits | 133 | IBM Heron-class device size |
| 1-qubit / 2-qubit gate error | 3e-4 / 3e-3 | typical published medians for current superconducting devices |
| Readout error | 1.5e-2 | typical published median |
| Shot overhead | 250 µs | default repetition delay on IBM devices |
| Layer time | 0.66 µs | two-qubit gate duration scale |
| Job overhead | 3 s | compile, load and return per job |
| Queue | 120 s mean (exponential) | public queues range from seconds to hours; scenarios vary this |
| Price | $0.30 per task + $0.00145 per shot | Amazon Braket per-task / per-shot pricing for superconducting QPUs |
| System power while running | 25 kW | commonly quoted wall-plug figure for a dilution-refrigerated system |
| Cloud state-vector simulator | ≤ 34 qubits, $0.075/min | managed simulator class (e.g. Braket SV1) |

Circuit fidelity is estimated as `(1-p1)^n1q · (1-p2)^(2·n2q) · (1-readout)^n`. Two-qubit counts for QAOA include
a routing factor `1 + 0.25 × average degree` for SWAPs when a dense graph is mapped to sparse hardware coupling.

## Result quality

* **MaxCut**: quality = best cut found / optimum. QAOA and random-sampling curves are fitted from E2
  (`results/calibration.json`); noise mixes the QAOA curve toward random sampling in proportion to `1 - fidelity`.
* **Kernel classification**: quality = test accuracy, measured in E2 for the quantum kernel and RBF SVM.
  The quantum feature map uses bandwidth `1/d` (Shaydulin & Wild, 2022), which we found necessary to beat
  chance at all.
* **Quantum simulation (transverse-field Ising chain)**: classical state vector is exact while it fits in
  memory (≤ 64 GB, about 32 qubits). Beyond that we model a matrix-product-state simulation with bond
  dimension capped at 1024 and entanglement growing by 0.5 bit per Trotter step; quality is
  `sqrt(chi_cap / chi_needed)` once the cap binds. This is the most assumption-heavy part of the model and the
  report says so. QPU quality is the circuit fidelity.

## What the scheduler can and cannot see

The scheduler sees the nominal figures above, slot queues on every resource, ping RTT, and the queue estimate
a provider publishes. It does not see true bandwidth, thermal throttling, or the real QPU queue. In the
simulator those hidden values come from the scenario; the adaptive policy has to learn them from feedback.
