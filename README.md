# NLP Home Assignment 4

**CS5760 — Natural Language Processing | Fall 2026**  
**University of Central Missouri — Department of Computer Science & Cybersecurity**

A simple Python project that covers **text preprocessing, named entity recognition (NER), and scaled dot-product attention**, along with the five theory questions from Home Assignment 4.

**Text → Tokens → Named Entities → Attention**

## Student information

| | |
|---|---|
| **Name** | Srinivas Reddy Pittu |
| **Course** | CS5760 — Natural Language Processing |
| **Semester** | Fall 2026 |
| **Assignment** | Home Assignment 4 |
| **University** | University of Central Missouri |

## Assignment overview

The assignment has two parts:

- **Part I — Short answers (Q1–Q5):** Sequence-to-sequence models, attention mechanisms, Transformer self-attention, multi-head attention, and Transformer architecture.
- **Part II — Programming (Q1–Q3):** Implement preprocessing with NLTK, identify entities with spaCy, and calculate scaled dot-product attention with NumPy. Each programming question also has two short-answer questions.

The full answers and calculations are in [the completed Word assignment](assignment/Home_Assignment_4_Completed.docx), with a text version in [answers.md](assignment/answers.md).

### Part I: theory covered

| Question | Topic | Main point |
|---|---|---|
| Q1 | Seq2Seq models and attention | The encoder represents the source sentence; attention helps the decoder use all encoder states rather than one fixed-length summary. |
| Q2 | Hard and soft attention | Soft attention uses differentiable weights. For scores `[1, 0, 1]`, the weights are `[0.422, 0.155, 0.422]`, giving context `[0.845, 0.578]`. |
| Q3 | Self-attention | Query, Key, and Value help tokens attend to related words, including distant words. |
| Q4 | Multi-head attention | Multiple heads learn different relationships; with 512 dimensions and 8 heads, each head uses **64 dimensions**. |
| Q5 | Transformer architecture | Positional encodings provide word order, masks hide future tokens, and Transformers parallelize training more easily than RNNs. |

## Part II: Python programs

### Q1 — NLP preprocessing

**File:** [`src/q1_preprocessing.py`](src/q1_preprocessing.py)  
**Library:** NLTK

The program processes this sentence:

> NLP techniques are used in virtual assistants like Alexa and Siri.

It performs three steps:

1. **Tokenization:** `word_tokenize()` separates the words and punctuation.
2. **Stopword removal:** removes common words such as `are`, `in`, and `and`, and excludes punctuation. The script uses the NLTK stopword corpus when installed or a small fallback list otherwise.
3. **Stemming:** `PorterStemmer()` reduces words to shorter stems, such as `techniques → techniqu` and `assistants → assist`.

**Output:**

```text
Original Tokens: ['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']
Tokens Without Stopwords: ['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri']
Stemmed Words: ['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']
```

The related short answers explain **stemming vs. lemmatization** and why stopword removal can help some tasks but harm tasks where words such as *not* matter.

### Q2 — Named Entity Recognition (NER)

**File:** [`src/q2_named_entity_recognition.py`](src/q2_named_entity_recognition.py)  
**Library:** spaCy (`en_core_web_sm`)

The program loads spaCy's English model and finds named entities in the assignment's LeBron James sentence. For every detected entity, it prints the **text**, **entity label**, **start character**, and **end character**.

**Observed output from the completed assignment:**

| Entity | Label | Start | End |
|---|---|---:|---:|
| LeBron James | PERSON | 1 | 13 |
| NBA | ORG | 21 | 24 |
| the United States | GPE | 35 | 52 |
| Laker | GPE | 56 | 61 |
| MVP | ORG | 84 | 87 |
| 2020 | DATE | 91 | 95 |

**Note:** The input begins with a space, so the first entity starts at index `1`. Python/spaCy end positions are **exclusive**. Entity labels can vary by model version; `Laker → GPE` is the model's observed prediction, not a manually assigned label.

The related short answers compare **NER with POS tagging** and discuss NER applications in **financial news and search engines**.

### Q3 — Scaled dot-product attention

**File:** [`src/q3_scaled_dot_product_attention.py`](src/q3_scaled_dot_product_attention.py)  
**Library:** NumPy

The program follows the four steps from the assignment:

1. Calculate similarity scores using `Q × K.T`.
2. Divide the scores by `√d`, where `d` is the key dimension.
3. Apply **softmax** along each row to get attention weights.
4. Multiply the weights by `V` to get the final output.

It uses the exact `Q`, `K`, and `V` arrays given in the Word assignment.

**Output (rounded to three decimal places):**

```text
Attention Weights:
 [[0.731 0.269]
 [0.269 0.731]]
Final Output Matrix:
 [[2.076 3.076 4.076 5.076]
 [3.924 4.924 5.924 6.924]]
```

The related short answers explain **why scaling by `√d` matters** and **how self-attention captures relationships between words**.

## Repository files

```text
nlp-from-tokens-to-attention/
├── README.md
├── requirements.txt
├── .gitignore
├── assignment/
│   ├── Home_Assignment_4_Completed.docx
│   └── answers.md
├── src/
│   ├── q1_preprocessing.py
│   ├── q2_named_entity_recognition.py
│   └── q3_scaled_dot_product_attention.py
├── notebooks/
│   └── Home_Assignment_4.ipynb
├── tests/
│   └── test_attention.py
└── outputs/
    ├── q1_sample_output.txt
    ├── q2_observed_output.txt
    ├── q2_model_note.txt
    └── q3_sample_output.txt
```

**What the coding/support files do:**

| File | Explanation |
|---|---|
| [`q1_preprocessing.py`](src/q1_preprocessing.py) | Defines `preprocess(sentence)`; prints original tokens, filtered words, and Porter stems. |
| [`q2_named_entity_recognition.py`](src/q2_named_entity_recognition.py) | Defines `show_entities(sentence)`; extracts and prints spaCy entity labels and character offsets. |
| [`q3_scaled_dot_product_attention.py`](src/q3_scaled_dot_product_attention.py) | Defines `scaled_dot_product_attention(Q, K, V)`; returns attention weights and the weighted output. |
| [`Home_Assignment_4.ipynb`](notebooks/Home_Assignment_4.ipynb) | Notebook version of Q1–Q3, with code and short answers organized by question; run cells in order. |
| [`test_attention.py`](tests/test_attention.py) | Checks that the Q3 attention weights and output match the expected numerical results. |
| [`requirements.txt`](requirements.txt) | Lists the Python packages needed to run the programs. |
| [`outputs/`](outputs/) | Stores reference outputs for comparison, including the observed spaCy run. |

## How to run

**1. Install the requirements** (Python 3.10 or newer recommended):

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```

The NLTK stopword download is optional because Q1 has a fallback list; the spaCy model installation is required for Q2.

**2. Run each question** from the repository's main folder:

```bash
python src/q1_preprocessing.py
python src/q2_named_entity_recognition.py
python src/q3_scaled_dot_product_attention.py
```

**3. Run the Q3 check** (optional):

```bash
python -m unittest discover -s tests -v
```

You can also open [`Home_Assignment_4.ipynb`](notebooks/Home_Assignment_4.ipynb) in Jupyter Notebook, VS Code, or Google Colab and run its cells in order after installing the dependencies.

## Submission

The Python files contain comments explaining the main steps. Upload the repository contents to **GitHub** and submit the **GitHub repository URL on Brightspace**, as required in the assignment instructions.

---

*CS5760 · Home Assignment 4 · Fall 2026 · Srinivas Reddy Pittu*
