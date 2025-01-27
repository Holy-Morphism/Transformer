# Custom Transformer-Based Text Summarization

## Project Overview
This project is part of my NLP course assignment to build a transformer-based Large Language Model (LLM) from scratch for text summarization. The model is designed to summarize dialogues, though a decoder-only model would have been more suitable for the task. However, the assignment required implementing a full transformer architecture.

## Dataset
The model is trained on the text summarization dataset available at:
[Text Summarization with Large Language Models](https://www.kaggle.com/code/lusfernandotorres/text-summarization-with-large-language-models/input)

## Implementation Details
### Model Architecture
- **Type:** Abstractive Transformer Model
- **Layers:** 6 encoder and 6 decoder layers (similar to the original paper)
- **Attention Heads:** 8
- **Positional Encoding:** Implemented to preserve input sequence order
- **Sequence Length:** 128 tokens (due to memory constraints)
- **Vocabulary:** Created from the provided dataset only (to reduce computational burden)
- **Normalization:** Applied after each encoder and decoder stack for slight performance improvement
- **Tokenization:** Word-level tokenization
- **Batch Size:** 32

### Training Details
- **Loss Function:** Cross-entropy loss for sequence-to-sequence generation
- **Optimizer:** Adam optimizer with learning rate scheduling (warm-up and decay)
- **Early Stopping:** Implemented to prevent overfitting
- **Challenges:**
  - Limited computational resources led to a reduced sequence length (128 instead of 512)
  - Difficulty in capturing context for dialogues longer than 128 tokens

### Evaluation Metrics
The model was evaluated using:
- **ROUGE Scores:**
  - ROUGE-1: 0.3016
  - ROUGE-2: 0.0769
  - ROUGE-L: 0.2350
- **Qualitative Evaluation:**
  - **Relevance:** Performs well for sequences around 128 tokens but struggles with longer or extremely short dialogues.
  - **Coherence:** Inconsistent; sometimes logical, sometimes not.
  - **Conciseness:** Generally effective at capturing essential information.

## Results and Analysis
- The model's performance, measured via ROUGE scores, is approximately half of what state-of-the-art models (e.g., T5) achieved on news summarization tasks.
- Longer and shorter dialogues posed significant challenges in context comprehension.

## References
### Learning Resources Used:
- [Introduction to PyTorch | Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)
- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Transformer from Scratch - YouTube](https://youtu.be/ISNdQcPhsts)

### Related Work:
- [Automated News Summarization Using Transformers](https://arxiv.org/abs/2108.01064)

## Future Improvements
- Use of pre-trained embeddings to improve generalization
- Implementing better tokenization methods such as subword tokenization (e.g., BPE or WordPiece)
- Experimenting with longer sequence lengths with better resource optimization
- Exploring decoder-only models for better performance in dialogue summarization

## Conclusion
This project involved significant learning and effort, given the lack of prior experience with deep learning and PyTorch. Despite constraints, the model provided insightful results and highlighted areas for further exploration and improvement.

