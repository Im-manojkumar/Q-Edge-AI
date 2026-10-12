# Q-Edge AI: when should an AI workload use a quantum processor?

*Minor project report draft. Department of Artificial Intelligence and Data Science, Batch 20:
Manoj Kumar P, Ayaan Syed A.*

## Abstract

Q-Edge AI is a working prototype of an adaptive orchestrator that places each incoming AI workload on an
edge CPU, edge GPU, cloud CPU, cloud GPU, a quantum circuit simulator, or splits it so that only the circuit
part runs on a quantum processing unit (QPU). The framework runs the same circuits on four quantum software
stacks (Qiskit, PennyLane, Cirq, Amazon Braket) through one interface, reaches real IBM Quantum and Amazon
Braket hardware with the user's own credentials, and falls back to a calibrated IBM-device emulator otherwise.
We measured the quantum paths against classical baselines instead of assuming an advantage. On today's
hardware figures the answer to the project question is clear: **classical execution wins for every workload
we tested, and the orchestrator's main value is choosing well between edge and cloud.** Across a simulated
stream of mixed workloads Q-Edge meets 94% of deadlines against 90% for the best single-objective baseline,
while using 28% less edge energy and about 1/50th of its cloud spend. Quantum offloading becomes the right
choice only in a narrow, quantified region: wide, deep quantum-simulation circuits on hardware roughly 100x
less noisy than today's, or optimisation instances where QAOA would beat the classical heuristic by about 0.24
in approximation ratio at a few dollars per task. The framework detects both regimes automatically.
When a QPU is used, *which* QPU and which qubits matter: routing each circuit by per-qubit calibration data
cut the average output error by 37% against a fixed device (2.2x median reduction, up to 17x on small
circuits) across 16 circuits on eight IBM device calibrations, and avoided broken qubit links that the
default placement walked into.

## 1. Problem and approach

The question from the project proposal: *given an AI workload, when should it execute locally on the CPU/GPU,
when at the edge/cloud, and when should a portion be offloaded to a QPU?*

We treat this as a placement problem with five objectives (latency against the deadline, edge battery
energy, total energy, money, and result quality) and three kinds of hard constraint (memory, qubit count,
data privacy). Quantum resources are one more option the scheduler may pick, never a default.

```mermaid
flowchart LR
  W[Incoming workload] --> A[Workload analysis<br/>FLOPs, data, deadline,<br/>privacy, circuit profile]
  M[Resource monitor<br/>slot queues, RTT,<br/>provider QPU queue] --> D
  A --> D{Decision engine<br/>constraints, quantum-benefit gate,<br/>weighted score}
  D -->|classical| C[Edge CPU / Edge GPU /<br/>Cloud CPU / Cloud GPU]
  D -->|hybrid| H[Classical host runs optimiser / SVM]
  H <--> Q[Quantum simulator or QPU<br/>Qiskit, PennyLane, Cirq, Braket,<br/>IBM Quantum, Amazon Braket]
  C --> E[Measure latency, energy,<br/>cost, quality]
  H --> E
  E --> F[Feedback: learned correction<br/>per workload kind and placement] --> D
```

### Workloads

| Kind | Classical path | Quantum path | Quality measure |
|---|---|---|---|
| Inference (CNN batch) | edge/cloud CPU or GPU | none | accuracy |
| Training (fine-tuning) | edge/cloud CPU or GPU | none | accuracy |
| Optimisation (MaxCut) | exact enumeration ≤ 22 nodes, simulated annealing beyond | QAOA: classical optimiser + quantum circuit evaluations | best cut / optimum |
| Kernel classification | RBF-kernel SVM | quantum-kernel SVM (ZZ feature map, kernel estimated on the QPU) | test accuracy |
| Quantum simulation (transverse-field Ising chain) | exact state vector, or a bond-dimension-capped MPS beyond ~32 qubits | Trotter circuit run natively | 1 - error in magnetisation |

Only the last three have a quantum formulation; the first two are never sent to a QPU.

### Decision engine

For each candidate placement the cost model predicts latency (queue + transfer + compute, or queue +
per-round overhead + shots for quantum), edge energy (local compute plus radio), total energy (including a
QPU's 25 kW system draw while it runs), money (cloud per-second rates, QPU per-task and per-shot prices) and
quality. A hybrid placement must then pass the **quantum-benefit gate**: a quantum formulation exists, the
circuit fits the device, data may leave the edge, estimated circuit fidelity ≥ 0.05, predicted quality is at
least the best classical option's, and the deadline is met. Survivors are ranked by a weighted score; the
weights are user-adjustable in the dashboard. The adaptive variant then keeps an exponentially weighted
ratio of measured to predicted latency and quality per (workload kind, placement) and corrects later
estimates. All modelling assumptions are in `docs/assumptions.md`.

## 2. Experiments

| | Question | How |
|---|---|---|
| E1 | Do the four quantum stacks agree, and how fast are they? | 4 circuit families, 2 to 20 qubits, each on Qiskit Aer, PennyLane, Cirq and Braket; plus the same noise model in all four. |
| E2 | Does the quantum path actually help on these workloads? | QAOA vs exact/greedy/annealing on 18 random graphs (4 to 14 nodes); quantum vs RBF/logistic kernels on breast-cancer data (2 to 6 features, 4 seeds); Trotter circuits vs exact. Noiseless, NISQ-noise (p2 = 3e-3) and an IBM Torino emulator. Results calibrate the cost model. |
| E3 | Does Q-Edge place work better than simpler policies? | Discrete-event simulation, 800 mixed workloads × 3 seeds × 7 scenarios × 7 policies. The scheduler sees nominal resource figures; the simulator executes against hidden true ones. |
| E4 | Where exactly is the quantum boundary? | Decision maps over circuit size and depth for today's and better QPUs, and over a hypothetical QAOA advantage and QPU price. |
| E5 | Does the prototype run end to end? | Real payloads through the live orchestrator, and every quantum-capable job forced through classical, simulator and QPU-emulator paths. |
| E6 | Does routing by calibration data pay off? | 16 TFIM, QAOA and kernel circuits (4 to 12 qubits) on eight IBM devices: v1's fixed device and default qubits vs the device and qubits chosen from per-qubit calibration, scored against the ideal output. |

## 3. Results

### 3.1 One interface, four stacks (E1)

Through the shared circuit format all four SDKs produce the same distributions: up to 8 qubits the
total-variation distance to the exact state vector averages 0.03 for every stack, which is the shot-noise
level for 4000 shots, and the average-⟨Z⟩ error stays below 0.05 up to 20 qubits. With an identical
depolarising model translated into each SDK's own noise API, the noisy distributions agree with Qiskit's to
within 0.12 TV at 8 qubits (again shot-noise sized). This matters for the research comparison: differences
reported later are properties of the algorithms and hardware, not of the SDK.

Speed differs more than correctness, and the ranking flips with size. Qiskit Aer pays a fixed overhead of
about 0.12 s per call, so for small circuits PennyLane and Cirq are 5 to 10x faster; that is why the same
33-evaluation QAOA loop on 8 qubits took 0.39 s on PennyLane and 3.2 s on Qiskit (appendix A2). Past about
16 qubits the picture reverses: at 20 qubits Qiskit was fastest on the QAOA and Trotter circuits (0.36 s and
0.52 s per 4000-shot run), Cirq on GHZ and the kernel feature map (0.11 s, 0.24 s), and PennyLane's
`default.qubit` was 3 to 5x slower on the deeper circuits (1.8 to 2.0 s). Braket's local simulator sat in
between. So the simulator stack is itself a placement choice that depends on circuit size and on how many
small calls an algorithm makes; the dashboard exposes it.

![](figures/e1_stack_speed.png)

### 3.2 Quantum vs classical, measured (E2)

**MaxCut.** Taking the best of 1000 QAOA shots, QAOA "finds the optimum" on every graph up to 14 nodes,
noiseless, noisy or on the IBM emulator. So does sampling uniformly at random, to within 2%. At these sizes
1000 shots cover a large part of the 2^n solution space, so best-of-shots says nothing about the algorithm.
The informative number is the mean cut of a single sample: QAOA (p = 1) averages 0.74 of optimum against
0.65 for a random sample, closing about 39% of the gap at 4 nodes and 11% at 14 nodes. Exact enumeration
solves every instance in under 5 ms and simulated annealing matches it, so on these instances there is no
quality gap for QAOA to close. Calibrating the cost model with these measurements (instead of the
best-of-shots numbers) is what stops it from over-promising QAOA on large graphs.

![](figures/e2_maxcut.png)

**Kernel classification.** The quantum-kernel SVM reached 0.78 to 0.85 test accuracy against 0.89 to 0.91
for an RBF SVM and 0.92 to 0.94 for logistic regression, at every feature count, and took seconds instead of
milliseconds. Realistic gate noise cost it very little (±0.03), so the gap is the feature map, not the
hardware. Without bandwidth scaling the textbook ZZ feature map did worse still (0.60 at 4 features in a pilot run);
the 1/d scaling of Shaydulin & Wild (2022) is what makes it usable, and it still trails the classical kernel.

![](figures/e2_kernel.png)

**Quantum simulation.** Trotter circuits on the noiseless simulator reproduce the exact magnetisation to
within 0.01; with NISQ-level noise the error stays below 0.03 for chains up to 12 qubits and 10 steps. These
sizes are trivially exact on a laptop's state vector in milliseconds, so the quantum path only becomes
interesting at sizes no laptop can verify, which is what E4 models.

### 3.3 Placement policies (E3)

Mixed stream: 45% inference, 10% training, 20% optimisation, 15% kernel classification, 10% quantum
simulation, Poisson arrivals. Values averaged over three seeds of 800 workloads.

| Baseline scenario | Deadlines met | Mean latency | Edge energy / task | Cloud + QPU cost / task |
|---|---|---|---|---|
| Edge only | 38% | 3 696 s | 282 J | $0.000 |
| Cloud only | 51% | 714 s | 61 J | $0.001 |
| Quantum first (offload whenever possible) | 6% | 16 695 s | 324 J | $397 |
| Random | 43% | 6 528 s | 663 J | $111 |
| Latency-greedy | 90% | 23 s | 85 J | $0.111 |
| **Q-Edge (static)** | **94%** | 25 s | **61 J** | $0.002 |
| **Q-Edge (adaptive)** | **94%** | 25 s | **61 J** | $0.002 |

Neither edge-only nor cloud-only placement works for a mixed stream: the edge drowns in training jobs, the
cloud loses latency-critical inference to upload time. Latency-greedy placement is strong on latency but
ignores energy, money and quality. Q-Edge gives up 2 s of mean latency to meet more deadlines with 28% less
edge energy and 1/50th of the spend. "Quantum first" is the cautionary baseline: sending every
quantum-capable task to a QPU costs about $400 per task at current Braket-style per-shot prices (a 120-circuit
quantum kernel needs millions of shots), meets 6% of deadlines and lowers quality.

When reality departs from the nominal model the feedback loop pays off, modestly:

| Scenario (hidden from the scheduler) | Static: deadlines / mean latency / p95 | Adaptive: deadlines / mean latency / p95 |
|---|---|---|
| Network congested (uplink at 20%) | 85.4% / 121 s / 1 018 s | **86.6% / 101 s / 769 s** |
| Edge overloaded (4x load, throttled) | 75.0% / 84 s / 660 s | **75.6% / 80 s / 603 s** |
| Future QPU whose real queue is 50x the advertised one | 93.8% / 38 s / 239 s | **94.0% / 33 s / 214 s** |

In the last scenario the adaptive policy learned the hidden queue and cut its QPU offloads from 45 to 26
tasks. When nothing is hidden (baseline, free QPU tier, QPU busy but correctly reported) static and adaptive
are within noise of each other, as they should be.

![](figures/e3_policies_baseline.png)

![](figures/e3_scenarios.png)

### 3.4 Where the quantum boundary lies (E4)

**Quantum simulation.** With today's error rates (2-qubit 3e-3) Q-Edge never chooses the QPU anywhere in
the grid of 8 to 128 qubits and 2 to 40 Trotter steps: circuits shallow enough to survive the noise are also
shallow enough for a classical MPS simulation, and deep ones fail the fidelity gate. A 10x better QPU still
does not change this. At 100x lower error, which is early fault-tolerant territory rather than today's
hardware, the QPU is chosen for 36% of the grid: chains of 40 qubits or more run for more than 20 steps,
where classical simulation runs out of memory or accuracy. At 32 qubits a cloud state-vector simulator
is the better option, because the circuit still fits one.

![](figures/e4_qsim_map.png)

**Optimisation.** We asked how large a QAOA advantage would have to be before Q-Edge offloads, on a hard
instance class where the classical heuristic reaches only 0.70 of optimum, using the 100x-better QPU. At 30
nodes a quality edge of 0.06 is enough, but the work goes to a *simulator*, because 30 qubits still fit one.
At 60 nodes the QPU is chosen once the edge reaches about 0.24 and a task costs at most about $6; at
$25 a task it never is. At 100 nodes the routed circuit's fidelity falls to 0.22 and no edge in the sweep
justifies it.

![](figures/e4_qaoa_advantage.png)

### 3.5 The prototype end to end (E5)

The live orchestrator ran ten real workloads, choosing classical placements for all of them, then ran
every quantum-capable job through the forced simulator and QPU paths. All paths completed. QAOA on the IBM
Torino emulator matched the classical optimum on 6 and 10 node graphs (at that size best-of-shots always
does, see 3.2); a Trotter circuit on the emulator came within 0.05 of the exact magnetisation, against
0.01 on the noiseless simulator. The emulator's wall time (90 to 180 s for QAOA) is the cost of simulating
the device's noise on a laptop, not a QPU time, so latency rows for it are not comparable. Predicted and measured quality agreed to within the per-run noise of
the small test sets. The full table is in appendix A7.

### 3.6 Calibration-aware routing (E6)

v1 treated "the QPU" as one device with one set of error rates. Real providers expose many devices whose
median error rates differ by about 7x and change daily, and within one chip some qubits and couplers are far worse
than the median (IBM reports broken couplers with error 1.0). Q-Edge now reads every device's per-qubit
calibration (free: it uses no QPU time), estimates each circuit's success probability (ESP) on every device
and qubit placement, and routes to the best. Without an IBM account it uses calibration snapshots of eight
real devices (three Eagle and five Heron chips) and emulates the chosen device with its own noise model.

| | v1 (Torino, default qubits) | calibration-aware routing |
|---|---|---|
| Mean output error, 16 circuits | 0.095 | 0.059 (37% lower) |
| Median error reduction per circuit | | 2.2x (range 1.1x to 17x) |
| Picked the device that measured best | | 10 of 16; worst miss 0.009 fidelity |

Three findings stand out:

* **Device choice gives most of the gain.** The router picked the same device for every circuit (Aachen, the
  snapshot with the lowest error rates), and it was the measured best or within 0.009 of it. With live data,
  queues and daily calibration changes would make the choice move between devices. Keeping Torino but choosing its qubits
  by calibration helped 13 of 16 circuits, by less.
* **Calibration catches what averages hide.** On Sherbrooke, the default placement used a broken coupler in
  7 of 16 circuits, giving fidelities of 0.38 to 0.72. Calibration-aware placement reached 0.75 to 0.98 on
  the same device.
* **Static figures cannot choose a device at all.** v1's single catalogue number predicts the same fidelity on
  every device. Per-qubit ESP ranks devices for a given circuit with a Spearman correlation of 0.88 against
  measured fidelity. It is a ranking tool rather than a quality forecast: an error-free run is sufficient
  but not necessary for a good output, so measured fidelity sits above ESP.

Qiskit's own optimisation level 3 already places qubits using average error data; our extra search over up
to 4000 placements beat it in only 9 of 128 cases, by a tiny margin. The gain over v1 comes from choosing the
device and from using that noise-aware placement, not from a better placement algorithm. Every number here
comes from emulators built from the same calibration data the router reads, so this shows the routing works
as designed; live hardware adds crosstalk and drift between calibrations, which the trace logger
(`python -m qedge calibration log`) is built to measure.

![](figures/e6_calibration_choice.png)

![](figures/e6_calibration_prediction.png)

## 4. Answer to the project question

* **Run locally on the edge CPU/GPU** when the data are private, the deadline is tight relative to upload
  time, or the job is small: most inference, small optimisation and kernel jobs.
* **Send to the cloud CPU/GPU** when compute dominates transfer: training, large inference batches, and
  quantum-circuit simulation up to about 32 qubits.
* **Offload a portion to a QPU** only when all of these hold: the task has a quantum formulation, its circuit
  fits and keeps a usable fidelity on the device, the quantum result is predicted to beat the best classical
  result, and queue, price and energy still fit the deadline and budget. On today's hardware none of our
  workloads met that bar; E4 maps where future hardware would.
* **When a QPU is used, route by calibration.** Pick the device and qubits from today's per-qubit error data,
  not a fixed device: E6 shows a 37% lower average output error and avoids broken couplers.

That conclusion is the framework working as intended. The value of Q-Edge is that the decision is explicit,
measured and re-evaluated as hardware improves: changing three numbers in the configuration (error rate,
queue, price) moves the boundary, and the dashboard shows it moving.

## 5. Limitations

* Classical payloads run on one machine; cloud tiers add modelled network time rather than calling a cloud.
  Resource figures are assumptions listed in `docs/assumptions.md`.
* The IBM Quantum and Amazon Braket adapters follow the current SDKs but have not yet been run against a
  live account; all QPU results here come from the IBM Torino emulator. Running the E5 jobs with
  `QISKIT_IBM_TOKEN` set is the next step and fits within the free Open Plan.
* QAOA was run at p = 1 with 25 optimiser iterations; deeper circuits and better optimisers could close more of
  the gap to the optimum, though not past the exact and annealing baselines on these instance sizes.
* The MPS quality model for classical simulation beyond 32 qubits is the least-grounded assumption. It
  decides where the E4 boundary sits, so that boundary should be read as indicative.
* E6 judges routing on emulators built from the same calibration snapshots the router reads, so it cannot show
  crosstalk or drift. The snapshots are from different dates (2025 to 2026), so the comparison is between
  device calibrations as published, not a same-day comparison. Device queues are only known with a live
  account; without one the router ranks on error rates alone.
* Small test sets in the live runs (10 to 30 points) make single-run accuracies noisy; E2 is the place for
  statistical comparison.

## 6. How to reproduce

```bash
pip install -r requirements.txt
pytest                                   # 73 tests
python -m qedge experiments              # all six experiments, then REPORT.md
python -m qedge dashboard
```
