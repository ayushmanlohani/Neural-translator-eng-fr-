# 🇫🇷 English-to-French Neural Machine Translation

A sequence-to-sequence Transformer model built from scratch using PyTorch, capable of translating English sentences into French. This project demonstrates a deep understanding of the Transformer architecture, including custom implementations of Multi-Head Attention, Encoder-Decoder blocks, and positional embeddings.

> **⚠️ Important Note on Usage:**
> This model was trained on a dataset primarily consisting of **short to medium-length sentence pairs**. As a result, it is optimized for translating concise phrases and simple sentences.
>
> It is **not** designed for long paragraphs, complex compound sentences, or large blocks of text. Attempting to translate such inputs may result in repetitions, cut-offs, or hallucinations (nonsense output). For best results, please break down longer text into smaller, individual sentences.
> 
> My main focus was on the implementation and learning about how transformer actually works behind the screen.
> 
The project includes a full training pipeline and a deployed **Streamlit** web interface for real-time inference.

🔗 **[Live Demo App] (https://jenice-truing-unsightly.ngrok-free.dev/)** 

---

## 📂 Project Structure

* **`model.ipynb`**: The training notebook. Contains the complete data preprocessing pipeline, custom Transformer class definitions, training loop, validation logic, and metric evaluation.
* **`inference.ipynb`**: The deployment notebook. Loads the saved model weights (`model.pt`) and vocabularies, and launches a Streamlit web app via `pyngrok` for public access.
* **`model.pt`**: Saved state dictionary of the trained model (achieved lowest validation loss).
* **`eng_vocab.txt` / `fr_vocab.txt`**: Vocabulary mappings generated during training.

---

## 🧠 Model Architecture

This model is not a fine-tune of a pre-existing model (like BERT or GPT); it is a **custom implementation of the Transformer architecture** (Vaswani et al., 2017).

### Key Hyperparameters
| Parameter | Value | Reason |
| :--- | :--- | :--- |
| **Embedding Size** | 256 | Balanced trade-off between model capacity and training speed for this dataset size. |
| **Attention Heads** | 8 | Allows the model to attend to different parts of the sentence (e.g., grammar, semantic meaning) simultaneously. |
| **Encoder Layers** | 6 | Standard depth to capture complex linguistic patterns in the source text. |
| **Decoder Layers** | 6 | Split into **3 Pre-Cross Attention** and **3 Cross-Attention** layers to effectively generate target sequences. |
| **Max Sequence Length** | 128 | Sufficient to cover the vast majority of sentence lengths in the dataset. |
| **Dropout** | 0.1 | Applied to prevent overfitting during training. |
| **Optimizer** | AdamW | Selected for its efficiency in handling sparse gradients and weight decay. |

---

## 📊 Performance Metrics

The model was trained for **10 Epochs** on a dataset of **175,621 sentence pairs**.

### 1. Loss (CrossEntropy)
* **Final Training Loss:** `0.4860`
* **Best Validation Loss:** `0.8402`
* *Why this matters:* The low validation loss indicates the model is generalizing well and effectively predicting the next word in the French sequence without simply memorizing the training data.

### 2. BLEU Score
The BiLingual Evaluation Understudy (BLEU) score measures how close the model's output is to the human reference translation.
* **Raw BLEU-4:** `9.60`
* **Normalized BLEU-4:** `15.59` (Case-insensitive)
* *Interpretation:* A score of ~15.6 on a custom, scratch-trained model indicates functional translation capabilities for simple to moderate sentence structures.

---

## 🛠️ How It Works

### Data Processing
1.  **Tokenization:** Custom regex-based tokenizer to handle punctuation and case normalization.
2.  **Vocabulary Building:** Dynamic vocabulary creation filtering out rare words (min frequency = 2).
    * English Vocab Size: ~9,782
    * French Vocab Size: ~13,478
3.  **Padding:** Sequences are padded with `<pad>` tokens and marked with `<sos>` (Start of Sequence) and `<eos>` (End of Sequence).

### Deployment Pipeline
The application is deployed directly from Google Colab using a tunnel:
1.  **Streamlit:** Builds the interactive frontend (text box, translate button).
2.  **PyTorch:** Handles the inference logic, loading the trained `model.pt` to generate predictions.
3.  **PyNgrok:** Creates a secure tunnel from the Colab local runtime to the public internet, providing a shareable URL.

---

## 🚀 How to Run Locally (or in Colab)

### Prerequisites
* Python 3.8+
* PyTorch
* Streamlit
* PyNgrok

### Steps
1.  **Clone the Repo / Download Files:** Ensure you have `model.pt`, `eng_vocab.txt`, and `fr_vocab.txt`.
2.  **Install Dependencies:**
    ```bash
    pip install torch streamlit pyngrok
    ```
3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
4.  **Access:** Open the URL provided in the terminal (usually `localhost:8501`).

---

## 📝 Future Improvements
* **Beam Search:** Implementing Beam Search decoding instead of Greedy decoding to improve translation fluidity.
* **BPE Tokenization:** Switching from simple split-based tokenization to Byte Pair Encoding (BPE) to better handle unknown words (`<unk>`).
* **Larger Dataset:** Training on the full WMT Eng-Fra dataset for higher BLEU scores.

---

