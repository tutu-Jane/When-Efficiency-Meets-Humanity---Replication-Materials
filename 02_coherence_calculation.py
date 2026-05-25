"""
Compute c_v coherence for the 12-topic BERTopic solution.

Reproduces the topic coherence value reported in Section 3.2 of:
  [Author Name(s)] (2026). When Efficiency Meets Humanity: Tracing the
  Humanistic Turn in AI-in-Higher-Education Research through Computational
  Topic Modeling. Education and Information Technologies (under review).

INPUTS
------
data/ml_topics_terms.xlsx
    Twelve topics (T00-T11), each represented by its top 15 terms.
    Columns: Topic, Top Terms (comma-separated)

data/corpus_metadata.xlsx
    Cleaned corpus of 4,280 publications. The script uses the Title and
    Abstract columns to construct document texts.

OUTPUTS
-------
results/coherence_results.csv
    Per-topic c_v, u_mass, and c_npmi coherence scores plus summary statistics.

Expected output (matching Supplementary File S4):
    Mean c_v coherence: 0.54 (per-topic range: 0.33 to 0.68)

DEPENDENCIES
------------
pandas >= 2.0
gensim >= 4.3
nltk >= 3.8

USAGE
-----
    python scripts/02_coherence_calculation.py

Author: [Your Name]
License: MIT
"""

import re
import warnings
from pathlib import Path

import pandas as pd
import nltk
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')

# Download NLTK stopwords on first run
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

# ============================================================
# File paths (adjust if running from a different directory)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
TOPICS_FILE = REPO_ROOT / 'data' / 'ml_topics_terms.xlsx'
CORPUS_FILE = REPO_ROOT / 'data' / 'corpus_metadata.xlsx'
OUTPUT_FILE = REPO_ROOT / 'results' / 'coherence_results.csv'


# ============================================================
# Preprocessing
# ============================================================
def preprocess_doc(text, multiword_terms):
    """
    Preprocess a document for coherence calculation.

    Steps:
      1. Lowercase, strip non-alphanumeric characters
      2. Join multi-word terms with underscore so they match the topic
         representation produced by BERTopic's c-TF-IDF with bigrams
      3. Tokenize, remove stopwords, keep short tokens (e.g., 'ai', '19')
         since these appear in the topic representations
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Replace multi-word terms with underscore-joined version
    # Sort by length (longest first) to avoid partial matches
    for term in sorted(multiword_terms, key=len, reverse=True):
        joined = term.replace(' ', '_')
        text = re.sub(r'\b' + re.escape(term) + r'\b', joined, text)

    tokens = [t for t in text.split() if t not in STOPWORDS]
    return tokens


# ============================================================
# Main
# ============================================================
def main():
    print('Loading data...')
    topics_df = pd.read_excel(TOPICS_FILE)
    corpus_df = pd.read_excel(CORPUS_FILE)
    print(f'  Topics:    {len(topics_df)}')
    print(f'  Documents: {len(corpus_df)}')

    # ---- Parse topic terms ----
    topics_terms = []
    for _, row in topics_df.iterrows():
        terms = [t.strip() for t in row['Top Terms'].split(',')]
        topics_terms.append(terms)

    # ---- Collect multi-word topic terms ----
    multiword_terms = set()
    for terms in topics_terms:
        for term in terms:
            if ' ' in term:
                multiword_terms.add(term)
    print(f'  Multi-word terms in topics: {len(multiword_terms)}')

    # ---- Preprocess corpus ----
    print('Preprocessing corpus...')
    corpus_df['Text'] = (
        corpus_df['Title'].fillna('') + ' ' + corpus_df['Abstract'].fillna('')
    )
    tokenized_docs = [
        preprocess_doc(text, multiword_terms) for text in corpus_df['Text']
    ]
    tokenized_docs = [doc for doc in tokenized_docs if len(doc) > 0]
    print(f'  Tokenized documents: {len(tokenized_docs)}')

    # ---- Join multi-word terms in topic lists ----
    topics_terms_joined = [
        [t.replace(' ', '_') for t in terms] for terms in topics_terms
    ]

    # ---- Build dictionary ----
    dictionary = Dictionary(tokenized_docs)
    print(f'  Dictionary size: {len(dictionary)}')

    # ---- Verify coverage of topic terms in corpus ----
    all_topic_terms = set()
    for terms in topics_terms_joined:
        all_topic_terms.update(terms)
    missing = [t for t in all_topic_terms if t not in dictionary.token2id]
    print(f'  Topic terms in corpus: '
          f'{len(all_topic_terms) - len(missing)}/{len(all_topic_terms)}')
    if missing:
        print(f'  Missing terms: {missing}')

    # ---- Compute coherence ----
    print('\nComputing c_v coherence...')
    cm_cv = CoherenceModel(
        topics=topics_terms_joined,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )
    per_topic_cv = cm_cv.get_coherence_per_topic()
    mean_cv = cm_cv.get_coherence()

    print('Computing u_mass coherence...')
    cm_umass = CoherenceModel(
        topics=topics_terms_joined,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='u_mass'
    )
    per_topic_umass = cm_umass.get_coherence_per_topic()
    mean_umass = cm_umass.get_coherence()

    print('Computing c_npmi coherence...')
    cm_npmi = CoherenceModel(
        topics=topics_terms_joined,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_npmi'
    )
    per_topic_npmi = cm_npmi.get_coherence_per_topic()
    mean_npmi = cm_npmi.get_coherence()

    # ---- Format output ----
    topic_labels = {
        'T00': 'AI in Higher Education (General)',
        'T01': 'Student Mental Health',
        'T02': 'ChatGPT Usage',
        'T03': 'Generative AI Tools in Education',
        'T04': 'GenAI Tools (Instrumental Use)',
        'T05': 'Language Learning and Engagement',
        'T06': 'Technology Acceptance and Adoption',
        'T07': 'Machine Learning and Learning Analytics',
        'T08': 'Medical Education and AI Attitudes',
        'T09': 'COVID-19 and Mental Health',
        'T10': 'Teacher AI Literacy and Self-Efficacy',
        'T11': 'Academic Integrity and GenAI'
    }

    results_df = pd.DataFrame({
        'Topic_ID':       [f'T{i:02d}' for i in range(12)],
        'Topic_Label':    [topic_labels[f'T{i:02d}'] for i in range(12)],
        'c_v':            [round(s, 4) for s in per_topic_cv],
        'u_mass':         [round(s, 4) for s in per_topic_umass],
        'c_npmi':         [round(s, 4) for s in per_topic_npmi],
    })

    # ---- Print results table ----
    print('\n' + '=' * 70)
    print('TOPIC COHERENCE RESULTS')
    print('=' * 70)
    print(results_df.to_string(index=False))
    print('-' * 70)
    print(f'MEAN    {"":<40}  c_v = {mean_cv:.4f}  '
          f'u_mass = {mean_umass:.4f}  c_npmi = {mean_npmi:.4f}')
    print('=' * 70)

    # ---- Save ----
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f'\nResults saved to: {OUTPUT_FILE}')

    # ---- Summary statistics ----
    print('\nSummary statistics for c_v:')
    print(f'  Mean:   {mean_cv:.4f}')
    print(f'  Median: {pd.Series(per_topic_cv).median():.4f}')
    print(f'  Min:    {min(per_topic_cv):.4f} '
          f'(T{per_topic_cv.index(min(per_topic_cv)):02d})')
    print(f'  Max:    {max(per_topic_cv):.4f} '
          f'(T{per_topic_cv.index(max(per_topic_cv)):02d})')


if __name__ == '__main__':
    main()
