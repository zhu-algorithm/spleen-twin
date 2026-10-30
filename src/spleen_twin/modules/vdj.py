"""V(D)J recombination — the naive B-cell receptor generator.

Mechanistic model of heavy- and light-chain rearrangement built on real human
germline segments and an empirically inferred generative model.

Germline segments and the recombination statistics (segment usage, exonuclease
trimming distributions, N-insertion length distributions and dinucleotide
Markov models) come from OLGA's ``human_B_heavy`` / ``human_B_kappa`` /
``human_B_lambda`` default models:

    Sethna Z, Elhanati Y, Callan CG, Walczak AM, Mora T.
    "OLGA: fast computation of generation probabilities of B- and T-cell
    receptor amino acid sequences and motifs." Bioinformatics 35(17):2974-2981
    (2019).

Biology represented here
------------------------
* RAG1/RAG2 cuts at recombination signal sequences; only the segment *choice*
  and the resulting junction are modelled, not the synapsis itself.
* Hairpin opening generates palindromic **P nucleotides**.  OLGA encodes these
  by pre-extending each segment's CDR3-proximal end with 6 nt of palindrome,
  so a deletion count ``d`` means ``max(0, P_LEN - d)`` P nucleotides survive
  and ``max(0, d - P_LEN)`` germline nucleotides are chewed back.
* TdT adds non-templated **N nucleotides** at the V-D and D-J junctions,
  sampled from the model's dinucleotide Markov chain.
* Out-of-frame joins and junctional stop codons produce **unproductive**
  rearrangements, which are retained so that downstream checkpoints
  (pre-BCR selection) operate on a realistic productive fraction.

Only ~1/3 of rearrangements are expected to be in frame by chance; the
pre-BCR checkpoint in :mod:`spleen_twin.modules.tolerance` consumes that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# Palindromic extension applied by OLGA to each CDR3-proximal segment end.
P_LEN = 6

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate(nt: str) -> str:
    """Translate a nucleotide string in frame 0; trailing partial codon dropped."""
    nt = nt.upper()
    return "".join(_CODON_TABLE.get(nt[i:i + 3], "X") for i in range(0, len(nt) - len(nt) % 3, 3))


def revcomp(seq: str) -> str:
    return seq.upper().translate(str.maketrans("ACGT", "TGCA"))[::-1]


@dataclass
class Junction:
    """Everything that happened between the germline segments."""

    del_v: int = 0
    del_d_left: int = 0
    del_d_right: int = 0
    del_j: int = 0
    p_v: int = 0
    p_d_left: int = 0
    p_d_right: int = 0
    p_j: int = 0
    trim_v: int = 0
    trim_d_left: int = 0
    trim_d_right: int = 0
    trim_j: int = 0
    n1: str = ""   # V-D insertion (TdT)
    n2: str = ""   # D-J insertion (TdT)

    @property
    def n_added(self) -> int:
        return len(self.n1) + len(self.n2)

    @property
    def p_added(self) -> int:
        return self.p_v + self.p_d_left + self.p_d_right + self.p_j


@dataclass
class Rearrangement:
    """A single rearranged chain."""

    locus: str                 # "IGH", "IGK" or "IGL"
    v_gene: str
    j_gene: str
    d_gene: Optional[str]
    junction: Junction
    cdr3_nt: str
    cdr3_aa: str
    seq_nt: str                # full V-REGION through end of J
    seq_aa: str
    productive: bool
    in_frame: bool
    has_stop: bool
    conserved_c: bool
    conserved_w_f: bool
    v_prefix_nt: int = 0       # length of FR1-CDR1-FR2-CDR2-FR3 (5' of CDR3)
    fr4_nt: int = 0            # length of FR4 (3' of CDR3)

    @property
    def cdr3_len_aa(self) -> int:
        return len(self.cdr3_aa)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["junction"] = asdict(self.junction)
        return d


class ChainRecombinator:
    """Samples rearrangements for one immunoglobulin locus."""

    #: conserved J-side residue terminating the CDR3 (IMGT junction convention)
    CONSERVED_J_RESIDUES = "FVW"

    def __init__(self, locus: str = "IGH", rng: Optional[np.random.Generator] = None):
        import olga.load_model as load_model
        import olga.sequence_generation as seq_gen

        model_dir = {
            "IGH": "human_B_heavy",
            "IGK": "human_B_kappa",
            "IGL": "human_B_lambda",
        }[locus]
        import olga
        md = os.path.join(os.path.dirname(olga.__file__), "default_models", model_dir)

        self.locus = locus
        self.has_d = locus == "IGH"
        self.rng = rng if rng is not None else np.random.default_rng()

        if self.has_d:
            self.genomic = load_model.GenomicDataVDJ()
            self.generative = load_model.GenerativeModelVDJ()
        else:
            self.genomic = load_model.GenomicDataVJ()
            self.generative = load_model.GenerativeModelVJ()

        self.genomic.load_igor_genomic_data(
            os.path.join(md, "model_params.txt"),
            os.path.join(md, "V_gene_CDR3_anchors.csv"),
            os.path.join(md, "J_gene_CDR3_anchors.csv"),
        )
        self.generative.load_and_process_igor_model(os.path.join(md, "model_marginals.txt"))
        self.sg = (seq_gen.SequenceGenerationVDJ if self.has_d else seq_gen.SequenceGenerationVJ)(
            self.generative, self.genomic
        )

        # Framework/CDR1/CDR2-bearing portion of each V gene, i.e. everything
        # 5' of the conserved cysteine that opens the CDR3.
        self._v_prefix = [
            full[: len(full) - len(cdr3_part)] for _, cdr3_part, full in self.genomic.genV
        ]
        # J sequence downstream of the conserved W/F that closes the CDR3 (FR4).
        self._j_suffix = [
            full[len(cdr3_part):] for _, cdr3_part, full in self.genomic.genJ
        ]

    # -- germline accessors -------------------------------------------------
    @property
    def v_genes(self) -> list[str]:
        return [g[0] for g in self.genomic.genV]

    @property
    def d_genes(self) -> list[str]:
        return [g[0] for g in self.genomic.genD] if self.has_d else []

    @property
    def j_genes(self) -> list[str]:
        return [g[0] for g in self.genomic.genJ]

    def v_prefix(self, v_index: int) -> str:
        """Germline FR1-CDR1-FR2-CDR2-FR3 nucleotides for a V allele."""
        return self._v_prefix[v_index]

    # -- sampling -----------------------------------------------------------
    def _rnd_insertion(self, length: int, C_R, C_first) -> str:
        if length <= 0:
            return ""
        num2nt = "ACGT"
        nt2num = {c: i for i, c in enumerate(num2nt)}
        seq = num2nt[int(C_first.searchsorted(self.rng.random()))]
        for _ in range(length - 1):
            seq += num2nt[int(C_R[nt2num[seq[-1]], :].searchsorted(self.rng.random()))]
        return seq

    def _draw_events(self) -> dict:
        """Draw recombination events using this object's RNG.

        Mirrors ``olga.sequence_generation.SequenceGeneration*.choose_random_recomb_events``
        but draws from ``self.rng`` so a campaign is reproducible from one seed.
        """
        sg = self.sg
        ev: dict = {}
        if self.has_d:
            # Heavy chain: V is marginal, (D, J) jointly distributed.
            ev["V"] = int(sg.CPV.searchsorted(self.rng.random()))
            dj = int(sg.CPDJ.searchsorted(self.rng.random()))
            ev["D"] = dj // sg.num_J_genes
            ev["J"] = dj % sg.num_J_genes
        else:
            # Light chain: (V, J) jointly distributed, no D segment.
            vj = int(sg.CPVJ.searchsorted(self.rng.random()))
            ev["V"] = vj // sg.num_J_genes
            ev["J"] = vj % sg.num_J_genes
        ev["delV"] = int(sg.given_V_CPdelV[ev["V"], :].searchsorted(self.rng.random()))
        ev["delJ"] = int(sg.given_J_CPdelJ[ev["J"], :].searchsorted(self.rng.random()))
        if self.has_d:
            dd = int(sg.given_D_CPdelDldelDr[ev["D"], :].searchsorted(self.rng.random()))
            ev["delDl"] = dd // sg.num_delDr_poss
            ev["delDr"] = dd % sg.num_delDr_poss
            ev["insVD"] = int(sg.CinsVD.searchsorted(self.rng.random()))
            ev["insDJ"] = int(sg.CinsDJ.searchsorted(self.rng.random()))
        else:
            ev["insVJ"] = int(sg.CPinsVJ.searchsorted(self.rng.random()))
        return ev

    def rearrange(self, max_tries: int = 200) -> Optional[Rearrangement]:
        """Attempt one rearrangement.

        Returns the rearrangement whether or not it is productive, or ``None``
        if the drawn deletions destroyed the CDR3 anchors (a geometrically
        impossible join, which the cell would simply not survive).
        """
        for _ in range(max_tries):
            ev = self._draw_events()
            v_seg = self.genomic.cutV_genomic_CDR3_segs[ev["V"]]
            j_seg = self.genomic.cutJ_genomic_CDR3_segs[ev["J"]]
            if len(v_seg) <= ev["delV"] or len(j_seg) <= ev["delJ"]:
                continue

            v_part = v_seg[: len(v_seg) - ev["delV"]]
            j_part = j_seg[ev["delJ"]:]

            jx = Junction(
                del_v=ev["delV"], del_j=ev["delJ"],
                p_v=max(0, P_LEN - ev["delV"]), p_j=max(0, P_LEN - ev["delJ"]),
                trim_v=max(0, ev["delV"] - P_LEN), trim_j=max(0, ev["delJ"] - P_LEN),
            )

            if self.has_d:
                d_seg = self.genomic.cutD_genomic_CDR3_segs[ev["D"]]
                if len(d_seg) < ev["delDl"] + ev["delDr"]:
                    continue
                d_part = d_seg[ev["delDl"]: len(d_seg) - ev["delDr"]]
                jx.del_d_left, jx.del_d_right = ev["delDl"], ev["delDr"]
                jx.p_d_left = max(0, P_LEN - ev["delDl"])
                jx.p_d_right = max(0, P_LEN - ev["delDr"])
                jx.trim_d_left = max(0, ev["delDl"] - P_LEN)
                jx.trim_d_right = max(0, ev["delDr"] - P_LEN)
                jx.n1 = self._rnd_insertion(ev["insVD"], self.sg.C_Rvd, self.sg.C_first_nt_bias_insVD)
                jx.n2 = self._rnd_insertion(ev["insDJ"], self.sg.C_Rdj, self.sg.C_first_nt_bias_insDJ)[::-1]
                cdr3_nt = v_part + jx.n1 + d_part + jx.n2 + j_part
                d_name = self.genomic.genD[ev["D"]][0]
            else:
                jx.n1 = self._rnd_insertion(ev["insVJ"], self.sg.C_Rvj, self.sg.C_first_nt_bias_insVJ)
                cdr3_nt = v_part + jx.n1 + j_part
                d_name = None

            v_prefix = self.v_prefix(ev["V"])
            j_suffix = self._j_suffix[ev["J"]]
            full_nt = v_prefix + cdr3_nt + j_suffix
            in_frame = len(cdr3_nt) % 3 == 0
            full_aa = translate(full_nt)
            cdr3_aa = translate(cdr3_nt) if in_frame else ""
            has_stop = "*" in full_aa
            conserved_c = bool(cdr3_aa) and cdr3_aa[0] == "C"
            conserved_wf = bool(cdr3_aa) and cdr3_aa[-1] in self.CONSERVED_J_RESIDUES

            return Rearrangement(
                locus=self.locus,
                v_gene=self.genomic.genV[ev["V"]][0],
                j_gene=self.genomic.genJ[ev["J"]][0],
                d_gene=d_name,
                junction=jx,
                cdr3_nt=cdr3_nt,
                cdr3_aa=cdr3_aa,
                seq_nt=full_nt,
                seq_aa=full_aa,
                productive=in_frame and not has_stop and conserved_c and conserved_wf,
                in_frame=in_frame,
                has_stop=has_stop,
                conserved_c=conserved_c,
                conserved_w_f=conserved_wf,
                v_prefix_nt=len(v_prefix),
                fr4_nt=len(j_suffix),
            )
        return None

    def rearrange_productive(self, max_attempts: int = 1000) -> Rearrangement:
        """Keep rearranging until a productive chain appears.

        Models the two chromosomal attempts plus receptor editing collapsed
        into a single "cell eventually succeeds or dies" outcome.
        """
        for _ in range(max_attempts):
            r = self.rearrange()
            if r is not None and r.productive:
                return r
        raise RuntimeError(f"no productive {self.locus} rearrangement in {max_attempts} attempts")


__all__ = [
    "P_LEN",
    "Junction",
    "Rearrangement",
    "ChainRecombinator",
    "translate",
    "revcomp",
]
