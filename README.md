# RAG (Retrieval-Augmented Generation)

This repository contains implementations and experiments with Retrieval-Augmented Generation (RAG) and its various types. RAG is a technique that combines information retrieval with generative models to improve the quality and relevance of generated responses, especially in tasks like question answering, summarization, and conversational AI.

## Project Structure

- `main.py`: Entry point for running RAG experiments.

- `InMemoryRAG/`: Contains core modules for RAG implementations.

  - `main.py`: Main logic for in-memory RAG.
  - `models.py`: Model definitions and utilities.
  - `prompts.py`: Prompt templates and management.
- `docs/`: Contains reference papers and documentation.
- `pyproject.toml`, `uv.lock`: Project dependencies and environment configuration.

## Types of RAG Implemented

- **In-Memory RAG**: Uses in-memory data structures for fast retrieval and generation.
- **Document RAG**: (Planned) Retrieval from a set of documents or PDFs.
- **Hybrid RAG**: (Planned) Combines multiple retrieval sources (e.g., in-memory + external DB).

## Getting Started

1. **Clone the repository**
   ```powershell
   git clone https://github.com/OmPatilGit/RAG.git
   cd RAG
   ```
2. **Install dependencies**
   ```powershell
   # If using uv
   uv sync
   ```
3. **Run the main script**
   ```powershell
   python main.py
   ```

## References
- [Attention Is All You Need (NIPS 2017)](docs/NIPS-2017-attention-is-all-you-need-Paper.pdf)
- [The Illusion of Thinking](docs/the-illusion-of-thinking.pdf)

## Contributing
Feel free to open issues or submit pull requests for new RAG types, improvements, or bug fixes.

## License
This project is licensed under the MIT License.
