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

## Job packing (E7, `qedge/quantum/packing.py`)

Several small circuits from different clients run side by side in one QPU job, sharing one queue wait, one
job overhead, one task fee and one set of shots (spatial multi-programming, after Das et al. 2019, QuCloud
2021 and Ohkura et al. 2022).

| Figure | Value | Basis |
|---|---|---|
| Placement | greedy, largest program first, each on its highest-ESP free qubits | same ESP terms as `ibm-auto`; exhaustive search per connected component of the routed program |
| Buffer | 1 coupling-map hop blocked around every program by default; 0 = programs may touch | adjacency counts every physical coupler, broken ones included |
| Capacity | first program that does not fit stops the allocation; reported, never dropped | greedy, so capacity is a lower bound on what the chip could hold |
| Crosstalk | **assumption**: factor c multiplies the 2q gate error on couplers that touch a qubit next to another program (buffer 0 only), added as extra depolarising error on top of the device noise | not measured; c = 1, 2, 5 are scenarios. Applied to every such gate, with no timing model (conservative). Murali et al. 2020 motivate it |
| QPU seconds per job | shots × (250 µs + depth × 0.66 µs), depth = deepest packed program | catalogue QPU figures; per program = job / k |
| Price per program | ($0.30 + shots × $0.00145) / k | Braket per-task / per-shot pricing applied to every device as an assumption (Braket does not sell IBM devices) |
| Free-plan throughput | 600 s / QPU seconds per program | IBM Open Plan: 10 QPU minutes per 28 days; IBM's metered usage also includes some per-job overhead, shown as a sensitivity column |

Emulation runs each program on its own region of the device's noise model. That is exact for Aer's device
noise models, whose errors are local to the qubits each gate touches, so programs on disjoint qubits have a
product output distribution (checked in `tests/test_packing.py`). The emulator has no crosstalk and no idle
noise of its own: a short program packed with a long one is not charged for any waiting. Whether it waits on
hardware, and in which state, depends on how the provider schedules the joint circuit; that is not modelled.
