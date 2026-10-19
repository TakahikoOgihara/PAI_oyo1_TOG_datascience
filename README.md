# ARM_BCRL006 — myCobot 280 Pi pick & place: **MuJoCo, on one PC, with the experiment that decides the design**

`12_ARM_BCRL005` (Isaac Lab `ManagerBasedRLMimicEnv` + NVIDIA's `isaaclab_mimic`, on a rented
GPU server) ported to **MuJoCo running locally**, and extended with the study the earlier
versions asserted but never measured:

> **Does MimicGen help? Does domain randomisation help? Do they compose?**
> 2 × 2, three seeds each, every policy evaluated on **both** a nominal and a randomised
> test bench.

The MDP is unchanged: same 31-dim observation in the same order, same 7-dim action, same
reward terms and weights, same success test, same virtual floor, same 20 Hz control. What
changed is the simulator, the machine it runs on, and the fact that the pipeline now ends
in a number instead of an opinion.

```
bash setup_env.sh && source ~/.venvs/arm_bcrl006/bin/activate
bash run_all.sh            # the 11-phase pipeline          (~1 h)
bash run_ablation.sh       # the 2x2 study                  (~80 min)
python src/record_video.py # the submission video           (~4 min)
python mujoco_sim/train_arm_mujoco.py --mode play           # watch it live
```

Everything needed to run it is below; the reason behind each choice is in the comments of
the file that implements it, and in the Hydra configs under `configs/`.

---

## The pipeline (11 phases, one command)

```
0  assets      myCobot 280 Pi -> MJCF, generated from deploy/raspi/mycobot_fk.py and
               cross-checked against MuJoCo's own FK to 1e-16 m. No download, no meshes.
1  collect     scripted IK expert generates demonstrations, wrist camera recorded,
               sharded across processes
               ★ this is the MimicGen SOURCE, and it is replaceable by ~10 real demos
1.5 mimicgen   OFF BY DEFAULT (mimicgen.enable=true turns it on -- see below).
               annotate -> warp each object-centric subtask onto a new scene -> replay
               -> keep only what the environment's own success term accepts
2  preprocess  outlier clip + robomimic HDF5 + ★IMAGE-QUALITY GATE (can stop the run)
3  BC          robomimic BC (low_dim) -> ONNX + MLflow Model Registry
4  eval        closed loop: random / scripted expert / BC
5  LeRobot     dataset export (optional, self-skips)
6  RL          stable-baselines3 PPO initialised FROM the BC weights, with the four
               stabilisers, closed-loop checkpoint selection and a regression guard
7  distill     the RL champion re-collects wrist-camera demos (privileged teacher ->
               sensor student)
8  visuomotor  ACT (default) / Diffusion Policy -> ONNX + Registry
9  eval        closed-loop evaluation of the deployed image policy
10 deploy      ★RELEASE GATE -> edge bundle (Jetson Orin Nano + Raspberry Pi)
```

**Two gates can stop the run, and that is the feature.** Phase 2 refuses a wrist-camera
dataset the policy cannot see through; phase 10 refuses to build an edge bundle for a
policy that does not clear its thresholds in simulation. `12_ARM_BCRL002` shipped without
either and went 0-for-10 on hardware.

---

## What ARM_BCRL006 changed

### 1. Isaac Lab / Isaac Sim → MuJoCo, and a GPU server → this laptop

Not a cost saving in the abstract: it is what made the experiment possible. On the server
the loop was *book, boot a 20 GB container, run, read, repeat*. Locally one condition of
the ablation costs about four minutes, so **twelve of them fit in 82 minutes** — and a
2 × 2 × 3-seed grid is a thing you actually run instead of a thing you would run if there
were time. Cycle time is a design parameter.

| | ARM_BCRL005 | ARM_BCRL006 |
|---|---|---|
| simulator | Isaac Sim 6.0 / Isaac Lab 3.0, PhysX | MuJoCo 3.x |
| hardware | rented GPU server + container | one PC (GPU used only for training) |
| robot asset | official URDF + COLLADA meshes → USD | MJCF generated from the FK constants |
| MimicGen | `isaaclab_mimic` (NVIDIA) | `mimicgen/generation.py` (~200 lines, eq. 1) |
| RL | RSL-RL | stable-baselines3 PPO |
| setup | container build + asset fetch | `bash setup_env.sh`, ~10 min |

### 2. The links near the base now render — because the failure mode is gone

ARM_BCRL005's viewer did not show the base, the first link or the flange. The cause was in
the assets: upstream COLLADA files declare `transparent` where the importer expects
`opaque`, and `joint7.dae` references a texture upstream does not publish, so the material
resolves to nothing. 005 patched both ends and still carried a placeholder-texture
workaround.

ARM_BCRL006 removes the class of defect instead of patching it again. The robot is emitted
as **primitive MJCF geometry** (capsules, boxes, cylinders) whose dimensions come from the
official joint origins in `deploy/raspi/mycobot_fk.py`. There is no material, no texture
and no transparency flag left to go wrong — and `assets/build_mycobot_mjcf.py --verify`
asserts MuJoCo's forward kinematics against that file to **1e-16 m**, so the simplification
cost no kinematic fidelity at all. What was traded away is photorealism, which this task
never used.

### 3. The 2 × 2 study

```
              MimicGen off        MimicGen on
DR off        A  baseline         B  mimicgen
DR on         C  dr               D  mimicgen + dr
```

Each condition is trained three times and each trained policy is evaluated on **both** test
benches — nominal and randomised. One test bench cannot separate "this made the policy
better" from "this made the test easier". Measured on a nominal bench alone, a DR-trained
policy usually looks *worse* than a baseline that overfitted to it, and a team that stops
there ships the overfitted one.

Measured (3 seeds, 256 closed-loop episodes per cell; random = 0.00, scripted
expert = 1.00):

| condition | train rows | nominal test | randomised test | gap |
|---|--:|--:|--:|--:|
| A neither | 13,093 | 0.001 ± 0.002 | 0.001 ± 0.002 | +0.000 |
| B MimicGen | 35,928 | **0.539 ± 0.236** | **0.453 ± 0.183** | −0.086 |
| C DR | 13,154 | 0.008 ± 0.008 | 0.013 ± 0.013 | +0.005 |
| D both | 36,454 | 0.436 ± 0.314 | 0.400 ± 0.262 | **−0.036** |

Four things fall out of it, and none of them were obvious in advance:

1. **MimicGen was decisive** (0.001 → 0.539). What bound this task was dataset
   *coverage*, not distribution width.
2. **DR alone bought nothing** (0.001 → 0.008). It does not add data; it widens
   the distribution over the same 170 demonstrations. Widening a distribution
   you cannot cover is dilution, not robustness.
3. **On top of MimicGen, DR does not raise the peak — but it buys exactly what
   DR is for.** Success 0.539 → 0.436 sits entirely inside the seed spread,
   while the robustness gap halves (−0.086 → −0.036) and placement error on the
   randomised bench improves from 57.7 mm to 43.5 mm. A binary success rate
   alone would have missed that.
4. **Seed variance exceeded the effect.** B = 0.223 / 0.603 / 0.791. A
   single-seed version of this study could have supported any conclusion,
   including the opposite one.

And the control that keeps the headline honest (`artifacts/ablation_matched/`):
collecting *2.8× more real expert demonstrations* instead of generating them
reaches **0.885 ± 0.082** at the same row count, against MimicGen's 0.539 ± 0.236.
(The best single seed hit 0.996, but point 4 above is precisely why that number
does not belong in a headline.) In simulation a demonstration is free, so
MimicGen loses that comparison — which is the point. On a real robot each
demonstration is a person-afternoon, and turning 10 of them into 400 usable
ones is the only version of this that exists.

And the data-matched control tells the other half of the story about DR: at
~37,000 rows it stops costing anything. Nominal 0.885 → 0.901, randomised
0.814 → 0.834 — and the **seed spread halves** (±0.082 → ±0.039). DR does not
raise the peak; it makes the result reproducible, which is a property you can
put in a delivery contract.

> **DR has to be paid for with data.** At 13,000 rows it buys nothing (0.008).
> At 37,000 rows it is free (0.901). The order is data first, distribution
> width second — never the reverse.

**The study changed the shipped defaults**, which is the only reason to run one:

| setting | 005 | 006 | why |
|---|--:|--:|---|
| `mimicgen.enable` | `true` | **`false`** | at equal rows, collecting beats generating here (0.885 vs 0.539), and generating made the result unstable (±0.236 vs ±0.082) |
| `collect.episodes` | 8 | **14** | ~20,000 rows sat below the data cliff; MimicGen had been papering over it |
| `domain_rand.enable` | `true` | `true` | free at this data budget, and its real beneficiary is hardware |

Phase 1.5 is not deleted — `mimicgen.enable=true` brings it back. MimicGen exists
for the real robot, where one demonstration costs a person an afternoon and
turning 10 into 400 is the only version of this that exists. The mechanism was
kept; the default was moved to match the measurement.

`run_ablation.sh` writes the full per-seed numbers, tables, plots and raw JSON to
`artifacts/ablation/` (`artifacts/` is regenerated and not versioned).

### 4. MLOps that decides something

Three tools, three jobs — and if you cannot say which question each one answers, you have
installed nothing:

* **MLflow** — the run ledger and the **Model Registry**. `src/infer.py` reads the
  `champion` alias to decide what ships, so MLflow is load-bearing, not reporting.
* **W&B** — the live view: PPO scalars per rollout, ACT loss per epoch, robomimic's BC
  curve replayed in after training, **and the closed-loop evaluation of every policy at
  inference time** as a table and bar chart. A training curve cannot tell a stakeholder
  whether the robot does the task.
* **TensorBoard** — the debug view: the per-iteration PPO scalars stable-baselines3 writes
  natively. When RL collapses this is the only one with the resolution to say why.

All three are local by default — SQLite, offline W&B, event files. No account, no API key,
no network.

---

## Inherited from `12_ARM_BCRL003` / `004` / `005`, unchanged

**The safety envelope.** `deploy/raspi/mycobot_fk.py` is the single, zero-dependency
definition of where the tool is and how low it reaches. The simulator conforms to *it*:
`arm_common/kinematics.py` imports the same constants, `assets/build_mycobot_mjcf.py`
builds the MJCF from them, `arm_common/ik.py` differentiates them, and the MuJoCo
environment enforces the same virtual floor with the same bisection the Raspberry Pi runs.
Start-up asserts prove all four agree. One source of truth ⇒ what the policy learns is
exactly what the hardware enforces.

**The three causes of ARM_BCRL002's 0-for-10 on hardware**, and their fixes:

| layer | what went wrong | what stops it now |
|---|---|---|
| perception | training images were blown out; the policy was effectively blind | phase-2 image-quality gate (exposure, contrast, red-block detection) |
| policy quality | a 0.55-success / 64 mm-error policy shipped unreviewed | phase-10 release gate + closed-loop checkpoint selection |
| control & safety | no floor limit, no contact stop, no homing | rate limit + virtual floor + latching impact stop on the Pi, trained against in sim |

**The scripted expert is the data source and the ceiling.** BC cannot exceed its
demonstrator, so phase 1 stops the pipeline if the expert's own success rate drops below
`mimicgen.min_source_success`.

---

## Repository map

```
assets/build_mycobot_mjcf.py   robot MJCF from the FK constants (+ --verify)
arm_common/                    expert phase machine, batched FK, analytic Jacobian, DLS IK
mujoco_sim/                    the environment, DR, policies, PPO, viewer, entry point
mimicgen/                      annotate (subtask segmentation) + generation (eq. 1)
bc_hdf5/                       robomimic dataset schema
arm_policy/                    ACT and Diffusion Policy (unchanged from 005)
scripts/                       BC and visuomotor trainers
src/                           the pipeline phases + ablation + video
deploy/raspi/                  the hardware safety layer and its FK (unchanged)
deploy/jetson/                 TensorRT runtime for the edge bundle (unchanged)
deploy/operator_pc/            browser operator console, relays to the Jetson (unchanged)
tests/                         26 tests, all CPU, all seconds
configs/                       one Hydra tree
```

`12_ARM_BCRL005` and earlier are referenced and copied from, never modified.
`artifacts/` starts empty; `run_all.sh` fills it.
