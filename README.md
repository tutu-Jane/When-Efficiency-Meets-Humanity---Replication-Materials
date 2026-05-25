# Replication Materials: When Efficiency Meets Humanity

Replication materials for the manuscript:

> [Author Name(s)]. (2026). When Efficiency Meets Humanity: Tracing the Humanistic Turn in AI-in-Higher-Education Research through Computational Topic Modeling. *Education and Information Technologies* (under review).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## Overview

This repository contains the data, lexicons, scripts, and results required to reproduce the findings reported in the manuscript. The study applies a multi-layer computational pipeline to 4,280 publications on artificial intelligence (AI) in higher education indexed in Scopus and Web of Science between 2000 and 2025.

**Theoretical framework.** Role Conflict Theory (Kahn et al., 1964) and Human Function Theory (Biesta, 2010, 2021) are integrated through a tension-based analytical framework comprising six dimensions: role conflict, role ambiguity, role overload, qualification, socialization, and subjectification.

**Computational pipeline.** BERTopic for topic modeling, theory-informed lexical mapping for topic-theory association, Correspondence Analysis and Cramér's V for structural analysis, and Piecewise OLS regression with joint F-tests for structural break detection. Quandt likelihood ratio (QLR) tests and Mann-Whitney U tests are reported as robustness checks.

**Key findings.**

1. The topic-theory matrix exhibits a strong structural divide (Cramér's V = 0.714).
2. HFT-aligned discourse has dominated the field throughout the observation period. Pre-2021 mean share was 0.18, post-2021 mean share was 0.26, with no statistically detectable structural break (F = 0.39, p = 0.684 at 2021).
3. RCT-aligned discourse, previously marginal (pre-2021 mean share = 0.009), exhibited statistically significant structural breaks at 2021 (F = 4.11, p = 0.032) and 2023 (F = 5.46, p = 0.013; R² = 0.50). Mann-Whitney U tests confirmed the asymmetry (U = 87.0, p = 0.001 for RCT; U = 70.0, p = 0.117 for HFT).

---

## Repository Structure

```
.
├── README.md                              [This file]
├── LICENSE-MIT                            [License for code]
├── LICENSE-CC-BY                          [License for data]
├── requirements.txt                       [Python dependencies]
│
├── data/
│   ├── corpus_metadata.xlsx               [4,280 publications: title, abstract, year, DOI, source, keywords, database]
│   ├── topic_indices.csv                  [Annual topic indices, 2000-2025, 26 rows]
│   ├── ml_topics_terms.xlsx               [BERTopic output: top terms per topic (T00-T11)]
│   └── ml_doc_assignments.xlsx            [BERTopic output: document-topic assignments]
│
├── lexicon/
│   └── theoretical_lexicon_120terms.xlsx  [6 theoretical dimensions × ~20 terms each]
│
├── expert_coding/
│   ├── expert_mapping_records.xlsx        [Individual rater judgments and consensus codes]
│   └── intercoder_reliability_summary.csv [κ and α calculation details]
│
├── scripts/
│   ├── 01_timeseries_analysis.py          [Piecewise OLS + QLR + Mann-Whitney U]
│   ├── 02_coherence_calculation.py        [c_v coherence computation]
│   └── README.md                          [Script documentation]
│
├── results/
│   ├── coherence_results.xlsx             [Supplementary File S4]
│   ├── time_series_analysis.xlsx          [Supplementary File S6]
│   └── figures/
│       ├── figure_5_time_series.png
│       ├── figure_5_time_series.pdf
│       └── [other figures]
│
└── supplementary/
    ├── S1_PRISMA_flow_and_keyword_scheme.pdf
    ├── S2_theoretical_lexicon.xlsx
    ├── S3_expert_coding_records.xlsx
    ├── S4_coherence_results.xlsx
    ├── S5_QLR_test_results.xlsx
    └── S6_time_series_analysis.xlsx
```

---

## Software Requirements

### Core dependencies

- Python 3.10 or later
- Operating system: any platform supporting the packages below (tested on Windows 11 and Ubuntu 22.04)

### Python packages

The complete list is provided in `requirements.txt`. Key dependencies:

| Package | Version | Purpose |
|---|---|---|
| bertopic | 0.16.4 | Topic modeling |
| sentence-transformers | 3.0+ | Document embedding (all-MiniLM-L6-v2) |
| umap-learn | 0.5+ | Dimensionality reduction |
| hdbscan | 0.8+ | Clustering |
| scikit-learn | 1.3+ | TF-IDF, stopword list |
| pandas | 2.0+ | Data handling |
| numpy | 1.24+ | Numerical computation |
| statsmodels | 0.14+ | OLS regression, F-tests |
| scipy | 1.11+ | Mann-Whitney U, F-distribution |
| gensim | 4.3+ | c_v coherence calculation |
| nltk | 3.8+ | Stopword list for coherence |
| matplotlib | 3.7+ | Figure generation |

### Installation

```bash
git clone https://github.com/[your-username]/[repository-name].git
cd [repository-name]
pip install -r requirements.txt
```

---

## Data Description

### `data/corpus_metadata.xlsx`

The cleaned corpus of 4,280 publications, exported from Scopus (n = 3,490) and Web of Science (n = 790) and deduplicated. Each row represents one publication.

| Column | Description |
|---|---|
| Title | Publication title |
| Abstract | Publication abstract |
| Year | Publication year (2000–2025) |
| DOI | Digital Object Identifier |
| Source | Journal or conference name |
| Keywords | Author-supplied keywords |
| Keywords_Plus | WoS-generated keywords (where applicable) |
| Database | Scopus or WoS |

Note: Only metadata is provided. Full-text PDFs are not redistributed; users seeking full text can retrieve articles from Scopus or Web of Science using the DOI.

### `data/topic_indices.csv`

Annual aggregated indices of HFT-aligned and RCT-aligned discourse, used for structural break analysis.

| Column | Description |
|---|---|
| year | Publication year |
| total | Total publications that year |
| H_index_share | Proportion of HFT-aligned content |
| E_index_share | Proportion of RCT-aligned content |
| Gap_share | H_index_share minus E_index_share |

### `data/ml_topics_terms.xlsx`

Output of BERTopic. Each of the twelve topics (T00–T11) is represented by its top fifteen weighted terms.

### `data/ml_doc_assignments.xlsx`

Output of BERTopic. Each document is assigned to one of the twelve topics. The full document-level assignments are too large to include directly; a representative sample is provided.

### `lexicon/theoretical_lexicon_120terms.xlsx`

The 120-term lexicon used for mapping topics to theoretical dimensions. Each term is labeled with:

- Its assigned dimension (one of six)
- Source category (canonical, recent literature, or corpus-informed expansion)
- Disambiguation notes for cross-dimensional ambiguous terms

### `expert_coding/expert_mapping_records.xlsx`

Records of three independent expert coders mapping the twelve topics to the six theoretical dimensions. Two columns are provided:

- `Mapped_Theory`: Individual rater's primary assignment before consensus moderation
- `Consensus_Buckets6`: Final multi-label consensus after moderated discussion

Discrepancies between these two columns reflect the normal coding process and were resolved through moderation. Reported κ and α values are calculated on individual ratings prior to consensus moderation.

---

## How to Reproduce

### Reproducing the time-series analysis (Section 4.3)

```bash
python scripts/01_timeseries_analysis.py
```

This script reproduces:

- Piecewise OLS regression for HFT and RCT discourse at break years 2021 and 2023
- Joint F-tests for structural break detection
- Mann-Whitney U tests for non-parametric confirmation
- Quandt likelihood ratio (QLR) test for robustness

Expected runtime: under 30 seconds on a standard laptop.

Expected outputs (matching Supplementary File S6):

```
HFT (Humanity) discourse, break at 2021: F(2, 20) = 0.39, p = 0.684, R² = 0.05
HFT (Humanity) discourse, break at 2023: F(2, 20) = 0.10, p = 0.907, R² = 0.02
RCT (Efficiency) discourse, break at 2021: F(2, 20) = 4.11, p = 0.032, R² = 0.46
RCT (Efficiency) discourse, break at 2023: F(2, 20) = 5.46, p = 0.013, R² = 0.50

Mann-Whitney U (post-2021 vs pre-2021):
  HFT discourse: U = 70.0, p = 0.117 (n.s.)
  RCT discourse: U = 87.0, p = 0.001 (significant)
```

### Reproducing the c_v coherence calculation

```bash
python scripts/02_coherence_calculation.py
```

This script computes c_v coherence (Röder et al., 2015) for the twelve-topic solution. Expected output:

```
Mean c_v coherence: 0.54 (per-topic range: 0.33 to 0.68)
```

Expected runtime: 1–2 minutes on a standard laptop.

### A note on BERTopic reproducibility

The trained BERTopic model file is not retained in this repository. Topic terms and document assignments are provided directly in `/data/`. Re-running BERTopic from scratch with identical default parameters may produce minor variations in topic boundaries due to the stochasticity of UMAP and HDBSCAN. The topic structure reported in the manuscript reflects the model trained in October 2025.

The BERTopic configuration used in the original analysis:

```python
from bertopic import BERTopic
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    nr_topics=12,
    calculate_probabilities=True
    # All other parameters used BERTopic library defaults:
    # UMAP: n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine"
    # HDBSCAN: min_cluster_size=10, metric="euclidean", cluster_selection_method="eom"
)
```

---

## File Cross-Reference with Manuscript

| Manuscript element | File in this repository |
|---|---|
| Section 3.1 corpus | `data/corpus_metadata.xlsx` |
| Section 3.2 BERTopic output | `data/ml_topics_terms.xlsx`, `data/ml_doc_assignments.xlsx` |
| Section 3.3 lexicon | `lexicon/theoretical_lexicon_120terms.xlsx` |
| Section 3.3 expert validation | `expert_coding/` |
| Section 4.3 time-series results | `results/time_series_analysis.xlsx` |
| Figure 5 | `results/figures/figure_5_time_series.png` |
| Supplementary File S4 (coherence) | `supplementary/S4_coherence_results.xlsx` |
| Supplementary File S6 (time series) | `supplementary/S6_time_series_analysis.xlsx` |

---

## Citation

If you use these replication materials, please cite the original manuscript:

```bibtex
@article{[lastname]2026humanistic,
  title   = {When Efficiency Meets Humanity: Tracing the Humanistic Turn in AI-in-Higher-Education Research through Computational Topic Modeling},
  author  = {[Author Name(s)]},
  journal = {Education and Information Technologies},
  year    = {2026},
  note    = {Under review}
}
```

And cite this repository:

```bibtex
@software{[lastname]2026replication,
  title     = {Replication Materials for "When Efficiency Meets Humanity"},
  author    = {[Author Name(s)]},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/[your-username]/[repository-name]}
}
```

---

## License

- **Code** (Python scripts in `/scripts/`): MIT License. See `LICENSE-MIT`.
- **Data and documentation** (everything else): Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE-CC-BY`.

You are free to use, modify, and redistribute these materials with appropriate attribution.

---

## Methodological Notes and Limitations

**Sample size.** The temporal series comprises 24 annual observations (2000–2025). This constrains the statistical power of structural break tests. Smaller breaks may not be detectable at conventional significance levels.

**ASR or text preprocessing dependencies.** The coherence calculation depends on the same text tokenization scheme used for the BERTopic input. Reproducing the exact c_v value requires identical preprocessing (stopword removal using the scikit-learn default English list, lowercasing, alphanumeric filtering).

**Replication of BERTopic.** The library uses non-deterministic dimensionality reduction (UMAP) and clustering (HDBSCAN). Setting a random seed is possible but not guaranteed to produce identical topics across different machine architectures or library versions.

**Scope.** The findings concern the discursive structure of a research field. They do not constitute claims about educational practice or institutional reform.

---

## Acknowledgments

The authors acknowledge the use of the BERTopic library (Grootendorst, 2022), Sentence-Transformers (Reimers & Gurevych, 2019), and the broader Python scientific computing ecosystem. Theoretical work on the integration of Role Conflict Theory and Human Function Theory builds on Kahn et al. (1964), Rizzo et al. (1970), and Biesta (2010, 2021).

---

## Contact

For questions about these materials or the underlying study:

- **[Author Name]**: [email@institution.edu]
- **Institution**: [Institution name, Department, Country]
- **Issues**: Please use the GitHub issue tracker for bug reports or reproducibility questions.

---

## Version History

| Version | Date | Notes |
|---|---|---|
| 1.0.0 | [Date of Zenodo archival] | Initial release for EAIT submission |

---

*This repository was prepared in accordance with the open science principles recommended by the FAIR data initiative (Wilkinson et al., 2016). Comments, corrections, and suggestions for improving reproducibility are welcome.*
