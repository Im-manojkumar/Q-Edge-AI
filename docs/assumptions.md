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

## Calibration-aware routing (`ibm-auto`)

| Figure | Value | Basis |
|---|---|---|
| Per-qubit and per-coupler errors | read from the device | IBM live calibration, or snapshots shipped with `qiskit-ibm-runtime` (dates in `python -m qedge calibration show`) |
| Unusable gate or readout | error ≥ 0.5 | IBM reports broken couplers with error 1.0 |
| Seconds per queued job | 20 s | converts IBM's "pending jobs" into a wait; public jobs vary widely |
| Layout search | Qiskit optimisation level 3, then up to 4000 more placements (8 shuffled VF2 restarts) scored by ESP | mapomatic approach (Nation & Treinish, 2023) |
| Region figures for the cost model | best connected n-qubit region found greedily from the 12 best couplers | fast approximation; the executor then scores the real circuit |

Estimated success probability (ESP) is the product of `(1 - error)` over every non-virtual gate and every
measurement on the chosen physical qubits. It predicts the chance of an error-free run, not the output
quality, so it is used to *rank* devices and layouts.

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

## Race mode (`submit(..., race=True)`)

| Figure | Value | Basis |
|---|---|---|
| Fallback raced against the QPU | lowest-score feasible classical placement in the decision | same score the policy ranks with; lowest predicted latency for policies that do not score |
| Deadline | `deadline_s`, else the workload's own deadline, wall clock from `submit()` | planning time counts against it |
| Cancellation granularity | one backend `run()` call | every `run()` checks the token first; an SDK call already in progress is not interrupted |
| Feedback to the adaptive policy | the delivered placement only, measured as outside a race | a cancelled QPU run only says "slower than the deadline" (censored), so it is not fed back |

* Local emulators cannot be stopped mid-job. After a cancel the abandoned thread finishes its current job in the
  background and its result is discarded. A QAOA run stops at its next `run()` call. `ibm-auto` also checks the
  token after its layout search, before it emulates or submits. A script exits only once these threads have ended.
* On live IBM hardware the Runtime job is registered on the token, and cancelling it calls `RuntimeJobV2.cancel()`.
  If IBM refuses (the job already finished), the result arrives and is discarded. This path follows the
  qiskit-ibm-runtime 0.50 API but has **not** been run against a live account.
* Both paths run on one machine here, so they compete for CPU and for Python's GIL. The emulator holds the GIL for
  long stretches, so the thread waiting for the deadline can wake late. The report records this as
  `overshoot_s`. Race timings are therefore not the timings either path would have on its own hardware.
* When the QPU wins, the classical path is not stopped (it is local and short here). Its result is recorded when
  it finishes.
