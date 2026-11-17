# CS5760 — Natural Language Processing | Home Assignment 4

This repository contains my **Home Assignment 4** answers and Python programs. All three programming questions are in **one Jupyter Notebook**: `Homework_4.ipynb`.

**Student:** Srinivas Reddy Pittu  
**ID:** 700777009

## Repository files

| File | Description |
| --- | --- |
| [Homework_4.ipynb](Homework_4.ipynb) | One notebook with Q1, Q2, Q3 code and saved outputs. |
| [Home_Assignment_4_Completed(4).docx](Home_Assignment_4.docx) | Completed written assignment, code, answers, and output screenshots. |
| README.md | Summary of the questions and explanation of the code. |

## Part I — Theory question summaries

**Q1. Sequence-to-Sequence Models and Attention**  
The encoder reads the input sentence, and the decoder generates the translation. Long sentences are hard to fit into one fixed-length vector, so attention helps the decoder focus on relevant words.

**Q2. Attention Mechanisms**  
Hard attention selects positions; soft attention assigns weights to all positions and can be trained using backpropagation. The given example produces scores **[1, 0, 1]**, weights **[0.422, 0.155, 0.422]**, and context vector **[0.845, 0.578]**.

**Q3. Self-Attention in Transformers**  
Self-attention connects words within the same sentence, even when they are far apart. **Q** searches for relevant information, **K** is used to match it, and **V** contains the information to combine.

**Q4. Multi-Head Attention**  
Multiple heads learn different word relationships. For **512 embedding dimensions** and **8 heads**, each head has **512 ÷ 8 = 64 dimensions**. The outputs are combined and projected.

**Q5. Transformer Architecture**  
Positional encodings provide word order; masked attention prevents access to future words. Transformers can train on tokens in parallel, unlike RNNs, but still generate text one token at a time.

### Q1. NLP Preprocessing — NLTK

**Question:** Tokenize the given sentence, remove stopwords, and apply stemming.

**Code explanation:** `word_tokenize()` separates words and punctuation. The code removes stopwords and punctuation, then `PorterStemmer()` reduces words to their stems. It prints all three stages.

**Final stemmed output:**

```text
['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']
```

**Short-answer summary:** Stemming reduces word forms; lemmatization uses vocabulary to find a base word (for example, *running* → *run*). Removing stopwords can help filtering, but removing *not* can change a sentence's meaning.

### Q2. Named Entity Recognition — spaCy

**Question:** Find named entities in the given LeBron James sentence, including their labels and character positions.

**Code explanation:** `spacy.load("en_core_web_sm")` loads the language model. `nlp(sentence)` processes the text, and `doc.ents` provides each entity's name, label, start position, and end position.

**Saved results:**

| Entity | Label | Start–End |
| --- | --- | --- |
| LeBron James | PERSON | 1–13 |
| NBA | ORG | 21–24 |
| the United States | GPE | 35–52 |
| Laker | GPE | 56–61 |
| MVP | ORG | 84–87 |
| 2020 | DATE | 91–95 |

*These are the saved spaCy predictions. `Laker` was labeled `GPE` by the model, although that is not the intended meaning in this sentence.*

**Short-answer summary:** NER finds names, places, and dates; POS tagging labels grammatical roles. NER is useful for financial news and search engines.

### Q3. Scaled Dot-Product Attention — NumPy

**Question:** Use the given **Q**, **K**, and **V** matrices to find attention weights and the final output.

**Code explanation:** The program calculates `Q @ K.T`, divides the scores by `sqrt(d)`, applies softmax to get weights, and multiplies those weights by `V` to get the result.

**Saved output (rounded to 3 decimals):**

```text
Attention Weights:
[[0.731 0.269]
 [0.269 0.731]]

Final Output Matrix:
[[2.076 3.076 4.076 5.076]
 [3.924 4.924 5.924 6.924]]
```

**Short-answer summary:** Scaling stops attention scores from becoming too large. Self-attention helps words use other words' context.

## How to run

1. Open `Homework_4.ipynb` in **Google Colab** or **Jupyter Notebook**.
2. Install the required libraries and spaCy model if needed:

   ```bash
   pip install nltk spacy numpy
   python -m spacy download en_core_web_sm
   ```

3. Run the three cells in order. Q1 uses a small backup stopword list if NLTK's stopword data is unavailable. To download the full list, run `import nltk; nltk.download('stopwords')` in a notebook cell.
