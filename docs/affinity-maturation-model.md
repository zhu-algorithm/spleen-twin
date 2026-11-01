# The affinity maturation model

How `spleen-twin` turns a germinal centre into an optimiser, written out as
equations. Implementation: `src/spleen_twin/modules/germinal_center.py`,
`shm.py`, `affinity.py`.

The germinal centre is a **mutation–selection loop with a moving target**. One
cycle is a dark-zone round of division and mutation followed by a light-zone
round of competition for a limited resource. Running it repeatedly is what
produces affinity maturation; nothing in the code optimises affinity directly.

---

## 1. State

Cell *i* at cycle *t* carries

| symbol | meaning |
|---|---|
| `s_i` | V-REGION nucleotide sequence (~380 nt), FR1–CDR3–FR4 |
| `r_i` | per-nucleotide IMGT region label |
| `M_i` | accumulated mutation list (position, from, to, region, silent?) |
| `x_i` | affinity, as `log10 K_D` in molar |
| `h_i` | antigen captured last cycle, in [0, 1] |
| `c_i` | clone identity, inherited from the founder |

The germinal centre also carries one scalar: `[Ag]_t`, the antigen held on
follicular dendritic cells.

---

## 2. Dark zone — divide

Help received translates into division number:

```
n_div(i) = d_min + round( (d_max − d_min) · clip(h_i, 0, 1) )
```

with `d_min = 1`, `d_max = 6`; founders enter with `h = 0.5`. This is the step
that converts last cycle's selection into this cycle's clonal expansion, and it
is why selection compounds instead of merely filtering.

The light zone can only process `L` cells, so the brood is scaled uniformly:

```
P  = Σ_i 2^n_div(i)
σ  = min(1, L / P)
children(i) ~ Poisson(2^n_div(i) · σ)      when the mean is < 1
            = round(2^n_div(i) · σ)         otherwise
```

Without `σ` the population would grow as 2^6 per cell per cycle.

---

## 3. Dark zone — mutate

Mutation count per child is Poisson in the total nucleotides replicated:

```
m ~ Poisson( μ · |s_i| · n_div(i) ),     μ = 1e-3 per base per division
```

Positions are drawn **without replacement**, weighted by the S5F five-mer
mutability of their sequence context:

```
P(mutate at j) ∝ M_S5F( s[j−2 … j+2] )
```

and the replacing base comes from the same five-mer's substitution
distribution, renormalised to exclude the current base:

```
P(b | j) = S_S5F( b | s[j−2 … j+2] ),   b ≠ s[j]
```

S5F is `HH_S5F` (Yaari et al., *Front Immunol* 4:358, 2013): 1024 five-mers
inferred from 806,860 silent mutations. **No CDR bias is added.** The familiar
CDR enrichment is produced downstream by selection; adding it to targeting as
well would double-count it. A child whose translation contains a stop codon
dies immediately.

Class switching rides along: an IgM/IgD child switches with probability 0.12
per cycle to an isotype drawn from the switch weights, irreversibly.

---

## 4. Score — sequence to affinity

Paratope = the three CDR loops, weighted `w = (0.6, 0.7, 1.8)` for CDR1/2/3,
reflecting CDR-H3's dominance of the contact surface. Complementarity is the
best register of the paratope against the epitope:

```
E = min over offsets o of  Σ_i w_i · ε( p_i , e_{o+i} )
```

with a pairwise term built from published amino-acid scales:

```
ε(a,b) = − w_hyd · max(0, J_a) · max(0, J_b)          hydrophobic burial (Janin)
         + w_ele · q_a q_b · (1.5 if same sign)        electrostatics at pH 7.4
         + w_ste · (V_a + V_b − 230)² / 2e4            steric fit
         − w_hb · ½ ( min(D_a,A_b) + min(D_b,A_a) )    hydrogen bonding
```

`E` has no absolute meaning, so it is calibrated on the **unmutated naive
repertoire** using two anchors:

```
z = (E − E_median) / (E_q0.1 − E_median)
```

`z = 0` at the median naive clone, `z = 1` at the best 1-in-1000 naive clone.
The map to affinity saturates, so each further unit of `z` buys less:

```
x = a − Δ·(1 − e^(−k·max(0,z)))  +  (floor − a)·(1 − e^(−k·max(0,−z)))
```

| constant | value | meaning |
|---|---|---|
| `a` | −6.3 | median naive clone, 5×10⁻⁷ M |
| `b` | −8.0 | best naive clone, 10⁻⁸ M |
| ceiling | −11.0 | attainable limit |
| floor | −4.5 | no complementarity |
| `Δ` | 4.7 | `a` − ceiling |
| `k` | 0.4490 | fixed by `Δ·(1 − e^(−k)) = a − b` |

Framework replacement mutations are charged a folding-stability cost:

```
x ← x + c · n_FR-replacement / (RT · ln10),   c = 0.18 kcal/mol
```

which is **0.1269 log units per framework replacement**. This is what stops the
landscape being climbed by wrecking the domain, and it is why framework
replacements get purged while CDR replacements accumulate.

> The oracle is a **fitness function, not a binding predictor**. It gives a
> smooth, epistatic, saturating landscape with the right dynamic range. It does
> not predict the K_D of any real antibody. `AffinityOracle` is the interface
> for substituting a structural method.

---

## 5. Light zone — capture antigen

A centrocyte strips antigen off the FDC in proportion to BCR occupancy
(Langmuir):

```
K_a,i = 10^(−x_i)
A_i   = K_a,i · [Ag]_t / ( 1 + K_a,i · [Ag]_t )
```

The half-occupancy point sits exactly at `K_D = [Ag]_t`. **This is the most
consequential line in the model.** Above it, capture saturates and clones stop
being distinguishable.

---

## 6. Light zone — compete for T-cell help

Help is the scarce resource. `S` help events are allocated by sampling
*without replacement* with probability proportional to captured antigen:

```
π_i = A_i^β / Σ_j A_j^β ,     β = 1.6
winners = S cells ~ sample-without-replacement(π)
```

Everything else apoptoses; winners carry `h_i ← A_i` into the next dark zone.
`S` follows the measured Tfh-to-naive-B ratio of 0.110 from the spleen atlas
(`AtlasReference.gc_parameters`), so the strength of competition is set by data
rather than tuned.

Selection is **relative**: a clone's fate depends on its competitors, not on
an absolute affinity threshold.

---

## 7. Exit and antigen consumption

```
recycle to dark zone   with probability ρ = 0.80
otherwise  memory      with probability f_mem · (1.6 if t ≤ 6 else 0.6),  f_mem = 0.45
           plasma      otherwise
```

Memory output is biased early and plasma output late, which is why memory
cells carry fewer mutations than plasma cells. The recycled pool is then culled
uniformly to carrying capacity `K`, and antigen is consumed:

```
[Ag]_{t+1} = max( [Ag]_min , γ · [Ag]_t ),    γ = 0.90
```

---

## 8. Why the antigen term is there

Define the selective advantage of a 10-fold affinity gain as
`d ln π / d(−x)`. From §5–§6:

```
d ln π / d(−x) = β · ln10 · (1 − A)
```

The gradient is proportional to **(1 − A)**: once a clone's capture saturates,
further affinity buys nothing and selection goes blind. Because the capture
midpoint tracks `[Ag]_t`, consuming antigen moves the window of
discriminable affinity to tighter values as the population improves.

Measured over a 30-cycle run from identical founders (seed 20261030):

| | selective advantage on the leading clone | |
|---|---|---|
| | cycle 1 | cycle 30 |
| antigen decays (γ = 0.90) | 2.93 | **3.18** |
| antigen constant (γ = 1.0) | 2.93 | **1.50** |

Constant antigen loses 49% of its selective pressure at the leading edge;
decaying antigen holds it. The outcome follows: the best clone reaches
**2.96e-09 M** with decay versus **6.85e-09 M** without, a 2.3-fold difference.

**The honest caveat:** decay helps the leading edge but not the bulk. In the
same run the population *median* ended at 2.03e-08 M with decay and
1.61e-08 M without — slightly better without. Antigen limitation sharpens
competition at the top while culling the middle harder. Which you want depends
on whether you are mining for the single best clone or for a broad panel.

---

## 9. What the loop produces

From 150 founders over 30 cycles, none of it imposed:

- population median affinity improves ~40-fold; the best clone reaches low nM
- replacement mutations accumulate ~3-fold faster per site in CDRs than in
  framework, purely from the stability penalty plus selection
- the polyclonal founder pool collapses to a handful of clones
- serum titre and serum *quality* rise together

## 10. Parameter summary

| parameter | symbol | value | source |
|---|---|---|---|
| SHM rate | μ | 1e-3 /base/division | canonical GC rate |
| mutation targeting | — | HH_S5F, 1024 five-mers | Yaari et al. 2013 |
| divisions per cycle | d_min, d_max | 1, 6 | GC dark-zone range |
| competition sharpness | β | 1.6 | model parameter |
| FDC antigen, initial | [Ag]_0 | 10 nM × uptake × retention | antigen dose |
| antigen decay | γ | 0.90 /cycle | model parameter |
| recycle probability | ρ | 0.80 | GC recycling literature |
| memory fraction of exits | f_mem | 0.45, time-biased | atlas-bounded |
| help slots | S | 0.110 × founders | CELLxGENE spleen atlas |
| framework stability cost | c | 0.18 kcal/mol | model parameter |
| naive median / best | a, b | −6.3, −8.0 | reported naive repertoire range |
| affinity ceiling | — | −11.0 | best-in-class therapeutic range |

Parameters marked "model parameter" are not measured; they are the knobs. The
rest are either measured here or taken from published values.
