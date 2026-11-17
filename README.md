# 📘 Natural Language Processing — Home Assignment 4

## 📝 Overview

This repository contains my **CS5760 Natural Language Processing — Home Assignment 4** for Fall 2026.

The assignment covers **Seq2Seq models, attention mechanisms, self-attention, multi-head attention, and Transformers**. It also includes three practical Python exercises using **NLTK, spaCy, and NumPy**.

All three programs and their saved outputs are in **one Jupyter notebook**: [`Homework_4.ipynb`](Homework_4.ipynb). The completed written answers are in the [`Home_Assignment_4.docx`](Home_Assignment_4.docx).

---

**Name:** Srinivas Reddy Pittu  
**Student ID:** 700777009  
**University:** University of Central Missouri  
**Course:** CS5760 — Natural Language Processing  
**Semester:** Fall 2026

---

## 📚 Part I — Theory Questions

### 1. Sequence-to-Sequence Models and Attention

- **Summary:** An encoder reads the input sentence, and a decoder generates the translated sentence.
- **Key idea:** A fixed-length context vector can lose information in long sentences. Attention lets the decoder use relevant encoder states at each step.

### 2. Attention Mechanisms

- **Summary:** Hard attention selects particular positions, while soft attention assigns weights to all positions. Soft attention is easier to train through backpropagation.
- **Calculation:** The given vectors produce attention scores **[1, 0, 1]**, softmax weights **[0.422, 0.155, 0.422]**, and context vector **[0.845, 0.578]**.

### 3. Self-Attention in Transformers

- **Summary:** Self-attention helps each word use information from other words in the same sentence, including distant words.
- **Key idea:** **Query (Q)** searches for matching information, **Key (K)** helps determine relevance, and **Value (V)** provides the information to combine.

### 4. Multi-Head Attention

- **Summary:** Multiple attention heads learn different relationships between words. Their results are combined and projected into the final representation.
- **Calculation:** With an embedding size of **512** and **8** heads, each head has **512 ÷ 8 = 64 dimensions**.

### 5. Transformer Architecture

- **Summary:** Positional encoding gives the model word-order information. Masked attention prevents the decoder from seeing future tokens.
- **Key idea:** Unlike RNNs, Transformers can process training tokens in parallel, although autoregressive text generation still happens one token at a time.

---

## 💻 Part II — Programming (One Notebook)

### 1. NLP Preprocessing — NLTK

**Goal:** Tokenize a sentence, remove stopwords, and apply stemming.

**Steps:**
- Use `word_tokenize()` to split the sentence into words and punctuation.
- Remove stopwords such as *are*, *in*, and *and*, along with punctuation.
- Use `PorterStemmer()` to reduce the remaining words to stems.
- Print the tokens, filtered tokens, and stemmed words.

**Key insight:** Preprocessing makes text easier to analyze. Stemming may produce shortened forms that are not complete dictionary words.

**Stemmed output:**

```text
['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']
```

**Short-answer note:** Lemmatization uses word knowledge to find a base form, while stemming uses simple word-reduction rules. Removing stopwords can be useful, but words like *not* may be important to meaning.

---

### 2. Named Entity Recognition — spaCy

**Goal:** Identify named entities in the provided LeBron James sentence and display their labels and character positions.

**Steps:**
- Load the English model using `spacy.load("en_core_web_sm")`.
- Process the sentence using `nlp(sentence)`.
- Use `doc.ents` to find entities and print their **text, label, start position, and end position**.

**Saved notebook results:**

| Entity | Label | Start–End |
| --- | --- | --- |
| LeBron James | PERSON | 1–13 |
| NBA | ORG | 21–24 |
| the United States | GPE | 35–52 |
| Laker | GPE | 56–61 |
| MVP | ORG | 84–87 |
| 2020 | DATE | 91–95 |

**Key insight:** NER recognizes names of people, organizations, places, and dates. The labels are model predictions; for example, spaCy predicted **GPE** for *Laker* in this run.

**Short-answer note:** NER identifies entities, while POS tagging identifies grammatical roles such as nouns and verbs. NER can help with financial-news analysis and search results.

---

### 3. Scaled Dot-Product Attention — NumPy

**Goal:** Calculate attention weights and the final output using the provided **Q, K, and V** matrices.

**Steps:**
- Multiply `Q` by `K.T` to calculate attention scores.
- Divide by `sqrt(d)` to scale the scores.
- Apply softmax to obtain attention weights.
- Multiply the weights by `V` to get the final output.

**Output (rounded to three decimals):**

```text
Attention Weights:
[[0.731 0.269]
 [0.269 0.731]]

Final Output Matrix:
[[2.076 3.076 4.076 5.076]
 [3.924 4.924 5.924 6.924]]
```

**Key insight:** Scaling keeps attention scores from becoming too large. Self-attention helps a model understand how different words relate to one another.

---

## 📂 Repository Files

| File | Description |
| --- | --- |
| [`Homework_4.ipynb`](Homework_4.ipynb) | **Single notebook** containing all three Python programs and their saved outputs. |
| [`Home_Assignment_4.docx`](Home_Assignment_4.docx) | Completed theory answers, coding questions, explanations, and output screenshots. |
| `README.md` | This overview of the assignment and code. |

## ▶️ How to Run

1. Open `Homework_4.ipynb` in **Jupyter Notebook** or **Google Colab**.
2. Install the necessary libraries and the spaCy English model if they are not already installed:

   ```bash
   pip install nltk spacy numpy
   python -m spacy download en_core_web_sm
   ```

3. Run the three notebook cells in order. For NLTK's full English stopword list, you can run `import nltk; nltk.download('stopwords')` in a notebook cell. The code also has a small built-in fallback list.

---

## 📌 About

A practical NLP assignment explaining **attention and Transformer concepts**, with Python examples of **text preprocessing**, **named entity recognition**, and **scaled dot-product attention**. The code and outputs are collected in one notebook, with full written answers in the accompanying Word document.
