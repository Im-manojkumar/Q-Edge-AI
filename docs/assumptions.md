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
| Crosstalk | **assumption**: factor c multiplies the 2q gate error on couplers that touch a qubit next to another program (buffer 0 only), added as extra depolarising error on top of the device noise | not measured; c = 1, 2, 5 are scenarios. Applied to every such gate, with no timing model (conservative about timing); ZZ coupling to a neighbour's idle qubits, 1q-gate and readout crosstalk are not modelled, so it is a scenario, not a bound. Murali et al. 2020 motivate it |
| QPU seconds per job | shots × (250 µs + depth × 0.66 µs), depth = deepest packed program | catalogue QPU figures; per program = job / k |
| Price per program | ($0.30 + shots × $0.00145) / k | Braket per-task / per-shot pricing applied to every device as an assumption (Braket does not sell IBM devices) |
| Free-plan throughput | 600 s / QPU seconds per program | IBM Open Plan: 10 QPU minutes per 28 days; IBM's metered usage also includes some per-job overhead, shown as a sensitivity column |

Emulation runs each program on its own region of the device's noise model. That is exact for Aer's device
noise models, whose errors are local to the qubits each gate touches, so programs on disjoint qubits have a
product output distribution (checked in `tests/test_packing.py`). The emulator has no crosstalk and no idle
noise of its own: a short program packed with a long one is not charged for any waiting. Whether it waits on
hardware, and in which state, depends on how the provider schedules the joint circuit; that is not modelled.

Fidelity loss in E7 is a difference of shot-based Hellinger fidelities, so it carries shot noise. The summary
reports `loss_se` (delta-method standard error, sqrt(F(1 - F)/shots) per fidelity, checked against 60 repeated
runs to be 0-30% on the large side) and `loss_z`; a loss with |z| < 2 is not resolved at 4000 shots. Identical
programs share one solo run, so for k copies of one circuit the solo term of the error does not shrink with k.

## QPU batching and race mode (E7)

These mechanisms live only in the discrete-event simulator; nothing here was run on a real QPU.

| Figure / rule | Value | Basis |
|---|---|---|
| Batch closes | at `max_tasks` (8) tasks, or `window_s` (5, 30 or 120 s) after its first task arrived | design choice; checked in arrival order, so an expired batch closes at its expiry time when the next task arrives |
| What one batch pays | one provider queue wait, one job overhead (3 s) and one per-task fee per round | a batch is submitted as a single job |
| Queue wait of a batch | the first member's own draw from the scenario's queue distribution | keeps the random-number stream identical to the unbatched run |
| Host share of a batched task | the classical host share (upload, optimiser set-up) runs while the batch is open; the job is submitted at batch close or when the last member's host share is done, whichever is later | the host work overlaps the batch window instead of adding to it, and one slow member delays the whole job |
| Multi-round programs (QAOA) | all rounds of a batch run in lockstep: one queue wait for round 1, then rounds back to back; each round is one job holding the members that still have rounds left | assumes a session-like reservation after the first job starts |
| Qubits a packed program needs | `n + ceil(0.25 n)` (25% idle buffer qubits around it) | placeholder for crosstalk isolation on heavy-hex chips; to be replaced by the packing experiment |
| Spatial packing | members are packed first-fit decreasing into groups that fit the device (133 qubits); a group runs side by side, groups run back to back | extension beyond the strict rule; `partial_packing=False` gives it: all members side by side, or all back to back. Usable qubits are the catalogue's 133, not a calibrated count that excludes broken qubits |
| Time and shots of a packed group | circuit j of every member runs in one combined circuit, which takes the most shots and the slowest per-shot time among the members that have a circuit j (for members of the same shape: the slowest member's time, one member's shots) | optimistic: assumes circuits of different members can be merged in one submission |
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
  Whether the QPU made the deadline is judged by when its thread finished, not by when the waiting thread woke,
  so a QPU result that lands in that gap is `late` and the classical result is delivered.
* The live log line is written at delivery. If the QPU thread is still running then (`quantum_settled: false`),
  the logged `quantum_status` is provisional; the final one is only on the in-memory `RaceInfo`.
* When the QPU wins, the classical path is not stopped (it is local and short here). Its result is recorded when
  it finishes.

## Split model (idea 3, `qedge/ml/`)

A small CNN backbone runs on the edge, a k-number embedding (k = 2..8) comes out of it, and a head turns it into
10 class scores: a linear layer, a parameter-matched MLP, or a k-qubit variational circuit with a linear readout
(the "quantum head", a dressed quantum circuit after Mari et al. 2020). The quantum head assists the classical
model; measured so far it is level with or below the classical heads (see `qedge/ml/__init__.py` and E8).

| Figure / rule | Value | Basis |
|---|---|---|
| Data | scikit-learn digits, 8x8, pixels / 16, stratified 60/20/20, seed 0 | small enough to train everything on a CPU; a toy next to a camera stream |
| Backbone and cut points | input (1x8x8 = 64 values) -> conv3x3 1->8, ReLU, maxpool -> conv1 (128) -> conv3x3 8->16, ReLU, maxpool -> conv2 (64) -> dense 64->k, tanh -> embedding (k) | conv1 is larger than the input, the classic Neurosurgeon observation (Kang et al., ASPLOS 2017) |
| Transmitted tensor | float32, 4 bytes per value | no compression or quantisation is modelled |
| FLOPs | 2 per multiply-add of conv and dense layers; bias, activations, pooling not counted | analytic, per sample |
| Quantum head | RY(pi/2 (z+1)) encoding, L x (RZ RY RZ + CNOT ring), <Z_i>, linear k -> 10 | L = 2 unless stated; `quantum_l0` (L = 0, encoding + readout only) is the ablation |
| MLP width | hidden units whose parameter count is closest to the quantum head's total | parameter-matched comparison; the quantum total includes k redundant angles, which favours the MLP slightly |
| Training | transfer mode: backbone + linear head end to end, then every head on the frozen embedding; Adam, softmax cross-entropy, early stopping on validation accuracy | validation only for selection; the test split is touched once per number |
| Attacker | honest-but-curious remote party with same-distribution labelled images (the training split) | ridge, one-hidden-layer MLP and kNN-regression decoders, hyperparameters chosen on validation; the strongest per metric. Numbers hold "against these attacks" only |
| Privacy rule (`private_ok`) | images re-identified that the label-only baseline does not (counted per image) <= 5 x chance, and r2 beyond the label <= 0.10 | policy thresholds, not a standard. Measured (seed 0, k = 2, 3, 4, 5, 6, 8): no cut passes, the embedding included, so private batches stay on the edge |
| Shared cuts in the profile | sizes and front FLOPs from one backbone per k; leakage = the worst over k; private only if private for every k | conservative. The file stores `back_flop` at k = 4 (and every k under `back_flop_by_k`); the router recomputes it per k as total(k) - front |
| Classical head of a k | linear or MLP, whichever has the higher validation accuracy (`classical_head`); the router costs and the executor runs that head (`head=best`) | picking on test accuracy would be model selection on the test split. The built-in placeholder profile names no head, so there `best` is the higher placeholder accuracy |
| Packed QPU cost of the head | `packing_cost`: catalogue QPU figures, Braket-style per-task / per-shot prices | same assumptions as E7 |

## Split placements: the router (`qedge/orchestrator/split.py`)

The router chooses where to cut the network (`edge_gpu>conv2>cloud_gpu`) and whether the head runs classically
or on a quantum resource (`edge_gpu>embedding>qpu`). It reads a *split profile*: `results/split_profile.json`
when E8 has written it (loaded by `Orchestrator()`, `qedge split profile` shows what is loaded), else built-in
defaults.

| Figure / rule | Value | Basis |
|---|---|---|
| `digits` defaults | bytes and FLOPs from the architecture; head accuracies 0.80 (linear, MLP), 0.75 (quantum), 0.70 (L = 0) and "not private" everywhere | **placeholders**, pessimistic round numbers so the router never favours the quantum head before E8 has measured it; a measured k replaces its entry, an unmeasured k keeps the placeholder and says so |
| `camera` profile | MobileNetV2 1.0 at 224x224 with a k-dim bottleneck: about 300 M multiply-adds per image (Sandler et al., 2018) minus the 1000-way classifier; activation sizes from the architecture; input 0.15 MB per image as in `make_inference` | **assumption profile**: nothing is measured on it and it has no trained model, so it can be planned but not run. Every cut but the embedding counts as not private; the embedding's accuracies and privacy verdict are inherited from digits and labelled so |
| FLOPs after a cut | total(k) - front FLOPs | a split neither adds nor loses work against the whole model |
| What comes back | float32 logits (4 x classes bytes per image) from a classical back; k expectation values (4 x k bytes) from the quantum head, whose linear readout runs on the edge | |
| Network | upload = batch x tensor bytes / uplink; a classical back adds its RTT; the quantum head's round trip is the per-round hop of the QPU model | same transfer model as every other kind |
| Quantum head on heavy-hex hardware | two-qubit gates scaled by a routing factor measured by transpiling the L = 2 ring onto FakeTorino (k = 3..8: 10/6, 17/8, 24/10, 33/12, 44/14, 40/16); each added gate adds one layer of depth | the L = 2 factor is used for every L; the depth rule is an upper bound when SWAPs on disjoint qubits run in parallel |
| Quantum head quality | f x (ideal quantum accuracy) + (1 - f) / classes, f = estimated circuit fidelity | a model: an error anywhere randomises the answer (pessimistic for errors that do not flip the argmax); shot noise is not modelled |
| Privacy | a placement is feasible for a private batch only if everything it transmits is at a cut marked `private_ok`; the raw input never is | replaces "raw data never leaves the edge" with a measured property against stated attacks |

## Running a split for real (`Executor`, `qedge/ml/zoo.py`)

| Figure / rule | Value | Basis |
|---|---|---|
| Model | the transfer-mode digits model for the workload's k, trained on first use with the default settings and cached in `results/models/` (not committed; a file whose training settings or source hash of `data.py`, `nets.py`, `qhead.py` differ from the code's is retrained; library upgrades are not detected, so delete the folder after one) | measured on this machine: k = 4 trains in 27.6 s (15.7 s of it the quantum head), k = 8 in 146 s (133 s); loading a cached model takes about 0.003 s |
| Images | `batch` test-split images drawn without replacement, seeded by the workload id | the quantum path and its race fallback see the same batch |
| Race with a split | the model is loaded (trained on a cold cache) before the race clock starts; the report counts what the QPU path sends off the edge even when the classical fallback is delivered | loading is setup, not part of the request; the QPU path starts in every race, so its embedding has left the edge |
| Forced placements | a forced split the cost model marks structurally invalid (front not on the edge, or back not a cloud classical resource) is refused, not run | what "left the edge" would mean nothing for it |
| Where it runs | every classical part runs on this machine; the edge/cloud split is honoured (the remote part starts from the float32 tensor the edge would send) and the bytes are counted, but the transfer is modelled, not sent | as for every other kind |
| Quantum head | `quantum_sim`: one circuit per image on the simulator stack; `qpu` on `ibm-auto` with batch > 1: packed, 8 images per job (idea 2's packing), else one circuit per image | hardware limits (`QEDGE_HW_MAX_CIRCUITS`, `QEDGE_HW_MAX_SHOTS`) apply before anything is submitted; the race token is checked before every job |
| QPU seconds and price in a result | `packing_cost` figures (catalogue QPU timings, Braket-style prices) | modelled, not metered; labelled in the result |
| Reported quality | batch accuracy, plus the agreement with the exact (noise-free, shot-free) quantum head on the same embeddings | a batch of a few images is a noisy estimate of the profile's test accuracy |
