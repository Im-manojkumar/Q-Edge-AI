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

## QPU batching and race mode (E7)

These mechanisms live only in the discrete-event simulator; nothing here was run on a real QPU.

| Figure / rule | Value | Basis |
|---|---|---|
| Batch closes | at `max_tasks` (8) tasks, or `window_s` (5, 30 or 120 s) after its first task arrived | design choice; checked in arrival order, so an expired batch closes at its expiry time when the next task arrives |
| What one batch pays | one provider queue wait, one job overhead (3 s) and one per-task fee per round | a batch is submitted as a single job |
| Queue wait of a batch | the first member's own draw from the scenario's queue distribution | keeps the random-number stream identical to the unbatched run |
| Multi-round programs (QAOA) | all rounds of a batch run in lockstep: one queue wait for round 1, then rounds back to back; each round is one job holding the members that still have rounds left | assumes a session-like reservation after the first job starts |
| Qubits a packed program needs | `n + ceil(0.25 n)` (25% idle buffer qubits around it) | placeholder for crosstalk isolation on heavy-hex chips; to be replaced by the packing experiment |
| Spatial packing | members are packed first-fit decreasing into groups that fit the device (133 qubits); a group runs side by side, groups run back to back | `partial_packing=False` gives the stricter rule: all members side by side, or all back to back |
| Time and shots of a packed group | the slowest member's execution time; the largest member's shot count, paid once (shots are shared) | optimistic: assumes circuits of different members can be aligned in one submission |
| Fidelity of a packed program | multiplied by `pack_fidelity_factor` (default 1.0, i.e. no crosstalk penalty) | not measured yet; the integrator sets it from the packing experiment |
| Splitting a batch's charges | QPU money in proportion to what each member would pay alone; QPU energy in proportion to its execution time | accounting convention; totals are unaffected |
| Batch window and the policy | the policy sees the expected wait for the batch to close as extra QPU wait (full window for a new batch) | `inform_policy=False` hides it; the adaptive feedback then learns it slowly |
| Feedback for batched tasks | given when the batch closes | delays learning by up to one window |
| Race-mode trigger | hedge when P(QPU queue > slack) > 0.05, slack = deadline minus everything except the provider queue | threshold is a parameter; E7 sweeps it |
| Queue model behind P(miss) | exponential with mean = provider estimate x learned bias | bias = censored exponential MLE of observed / published waits, starting at 1; a lognormal real queue breaks the exponential assumption on purpose in `future_qpu_heavy_tail` |
| Fallback | best-scoring classical placement predicted to meet the deadline; no hedge if there is none | a fallback expected to be late would only add cost |
| Which answer counts | the QPU answer if it arrives by the deadline, otherwise the classical one at its own latency | progressive delivery: the classical answer is handed over when ready and stands unless the QPU beats the deadline |
| Cancellation and charges | the QPU job is cancelled at the deadline; queued jobs that are cancelled cost nothing, jobs that started executing are charged in full; the classical fallback is always charged | assumption about provider billing (per-task/per-shot pricing charged on execution) |
| Feedback for a cancelled job | latency censored at the deadline; quality not observed (the prediction is fed back, which leaves the learned offset unchanged) | censoring biases the learned latency ratio low |

New scenarios (kept out of the E3 matrix): `future_qpu_heavy_tail` uses the `future_qpu` figures, but the true QPU
wait is lognormal with the same 12 s mean and sigma 1.5 (median 3.9 s; about 1% of jobs wait more than 10x the
mean, against 0.005% for an exponential queue). `quantum_heavy_future` uses the `future_qpu` figures, 3x the
arrival rate (0.6 tasks/s) and a mix of 40% quantum simulation, 30% optimisation, 10% kernel classification,
15% inference and 5% training.
