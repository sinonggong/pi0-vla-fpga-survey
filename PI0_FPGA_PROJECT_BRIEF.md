# Project brief: the whole pi0 VLA model on one Achronix Speedster7t FPGA

Purpose of this brief: the only description of our own project given to a writer who cannot see the source tree.
Every number is tagged [M] measured on the board / in cycle-accurate RTL simulation, or [P] projected by a model.
Numbers not in this brief must not be attributed to the project.

## 1. Target model and workload

* **Model:** pi0 (Physical Intelligence), a vision-language-action (VLA) policy: a PaliGemma VLM (SigLIP So400m/14
  vision encoder, 27 layers, width 1152, 16 heads of 72; Gemma-2B language model, 18 layers, width 2048, MLP 16384
  with GeGLU, 8 query heads of 256 with one shared KV head) plus a 300M-parameter "action expert" (Gemma-style, 18
  layers, width 1024, MLP 4096) that attends to the VLM's KV cache.  Actions are produced by flow matching:
  10 Euler integration steps of the action expert per inference, producing a chunk of 50 future actions.
* **Workload per inference ("one chunk")** as deployed: 2 camera images -> 2 x 256 SigLIP tokens; a 525-token
  prefix (image + language tokens) through the Gemma-2B prefix pass (KV cache); then the action expert on 51 tokens
  (state + 50 action tokens) for 10 steps, each step attending to the 525 prefix keys plus its own.  About 1.47 TMAC
  per chunk [P, from the GEMM list].  Very different shapes: prefix GEMMs have M ~ 525 rows (compute-bound, activation
  re-streaming dominates), expert GEMMs have M = 51 (weight-movement-bound).
* **Checkpoint / task:** pi0 fine-tuned on the DROID robot dataset; evaluation on 46 held-out recorded frames against
  the fp32 PyTorch reference, reported as the maximum and mean joint-angle error of the 7-DoF arm over the action chunk.

## 2. Platform

* **Device:** Achronix Speedster7t AC7t1500 on the VP815 PCIe card (PCIe Gen4 x16, 16 x 2 GB GDDR6 channels reached
  through a 2D network-on-chip (NoC); fabric logic enters the NoC through "NAP" access points).
* **Hard blocks used:** 2,560 MLP72 machine-learning blocks (each: 16 INT8 x INT8 MACs per cycle, or 32 signed
  INT4 x INT4 MACs per cycle in the SIGNED 4x4 mode, block-floating-point modes, an accumulator, and a hard cascade
  that forwards an operand word to the next MLP72 in the same column); 2,560 BRAM72K; LRAM2K small RAMs (each MLP72
  reserves its paired LRAM2K).  Cascade connectivity and multiplier modes are static bitstream parameters.
* **Tools:** Achronix ACE 10.5.2 with Synplify; Verilator for RTL simulation.
* **Reference point:** NVIDIA Jetson AGX Thor running the same pi0 chunk, compiled bf16: **125.3 ms** [M].

## 3. Architecture ("node array")

* **35 autonomous nodes**, each with its own NAP on the NoC and its own program fetched from GDDR6; nodes synchronise
  through flag words in GDDR6 (WAIT / POST commands).  The host only loads inputs, starts the array and reads results.
  * 24 integer GEMM chain nodes (12 with 32 multiplying stages, 12 with 16), 3 "PV" chain nodes for attention
    probability x value (uint8 x int8, 16 stages), 8 vector nodes (2 lanes each).
* **Column-parallel MLP72 chain:** a feeder MLP72 reads an activation word from its BRAM and forwards it up the hard
  cascade; every stage multiplies the same word by a different output column held in its own BRAM72K and accumulates
  its own column sum.  One row pass of W words yields d output columns; a tile covers d x floor(512 / W) columns; the
  row period is max(W + 1, d, ~14) array cycles.  All 16 (or 32 at INT4) multipliers of every stage are busy during
  a pass (the "sum the partial products up the cascade" alternative wastes 15/16 of them).
* **Vector nodes:** bf16 element operations with table-based transcendental functions (GELU, exp, rsqrt,
  sigmoid): LayerNorm / RMSNorm, softmax, GeGLU, RoPE, residual adds, per-token quantisation (row scales computed on the
  fly), dequantisation with per-row x per-column scales, fused quantise tails.
* **Numerics (accurate path):** W8A8 on every GEMM: weights per output channel (MSE clip), activations INT8 per token
  (dynamic), SmoothQuant on the language model, uint8 attention probabilities, bf16 at every vector interface.
  46-frame result on silicon: joint max 0.665 deg, mean 0.072 deg vs fp32 [M].
* **Compiler ("chunk generator"):** lowers the whole chunk to per-node programs: GEMM tiling per depth class, a striped
  GDDR6 address map, double-buffered feeders, weight-load hoisting above waits, fused quantisation, attention row
  blocks with layer overlap, pruning of dead work (the last prefix layer only needs its K/V), per-node load balancing.
  Its expected-memory model makes every silicon run checkable beat for beat ("bit-exact").
* **Clocks:** three domains -- array (MLP72 chains), vector, fabric (control, NoC side) -- written array / vector /
  fabric MHz.

## 4. Results on silicon (per chunk, full 35-node array, bit-exact against the compiler's expected memory) [M]

| design point | clocks (MHz) | chunk time | accuracy (46 frames) |
|---|---|---|---|
| first full-array chunk (09-23) | -- | 6.48 s | bit-exact |
| INT8, best schedule | 500 / 288.9 / 216.7 | **1.221 s** | joint max 0.665 deg |
| INT4 chains (W4A4 on every non-attention GEMM) | 437.5 / 279.6 / 244.6 | **1.007 s** | joint max 44.9 deg (not usable) |
| INT4 + GeGLU/GELU chain epilogue vs INT4 alone, same bitstream | 481.25 / 248.4 / 180.7 | 1.047 vs 1.115 s (**-6.1 %**) | identical to INT4 (bit-identical epilogue) |
| fabric-light chains (912 multiplying MLP72s) | 310 / 216.7 / 177.3 | 1.588 s | bit-exact |

* The schedule alone took the same INT8 hardware from 1.901 s to 1.221 s (memory layout, ordering / overlap, fusion,
  dead-work removal, then clocks) [M].
* Where the 1.221 s goes [P, calibrated model]: vector work 50 %, chain compute 25 %, serial weight loading 15 %,
  feeder stalls 6 %.  After INT4 about 82 % of the critical path is vector work [P].  Host overhead ~47 ms per call.
* Utilisation: the fabric (routing) is the binding resource, not the multipliers: a full array uses ~760-860 of the
  2,560 MLP72s at 63-75 % of the logic tiles; bigger arrays stop routing.

## 5. Techniques explored (status)

| technique | idea | status |
|---|---|---|
| chain result-path epilogues | dequant + GELU / GeGLU computed on the chain's int32 results before they leave the node, bit-identical to the vector ops; removes int32 round trips and vector passes | on silicon: -6.1 % [M] |
| K-split dequant epilogue | the chain combines the fp32 partial sums of K-split GEMMs (prefix down projection, 4 segments) itself, reading the previous partial from GDDR6 | RTL simulation: that vector stage -81 %, reduced chunk -6.4 % [M, sim]; build running |
| block-floating-point MLP72 chains | BFP8 operands with fp24 accumulation in the MLP72 | accuracy better than INT8 (0.70 vs 0.97 % action error) [M, software]; RTL sim only |
| fabric-light chain | weights loaded through the BRAM72K write cascade, results drained through the MLP72 output cascade in taps; ~2.4 vs 21.2 logic tiles per stage | on silicon (912 multiplying MLP72s) [M] |
| run-time fused chain pairs | two chains share activation rows over a link and act as one array twice as wide, chosen per GEMM at run time | RTL simulation bit-exact, read traffic -11..-23 % [M, sim]; ~0 latency gain expected [P]; build running |
| chain fission (32 -> 2 x 16 at run time) | a second, statically configured feeder in the middle of a cascade | legal and simulated; a run-time switch beats the best fixed shape by only 0.02-0.12 % [P] |
| INT4 (W4A4) | MLP72 SIGNED 4x4 mode | 1.007 s but unusable accuracy; 4-bit weights are destructive for this model [M] |

## 6. Honest limits

* One AC7t1500 remains ~8x slower than the Jetson Thor at matched accuracy (1.221 s vs 125.3 ms).
* The vector nodes and the serial weight loads dominate; the MLP72 array is under-utilised because routing, not
  multipliers, limits the array size.
* INT4 accuracy is not acceptable; latency results below 1.2 s rely on INT4.
