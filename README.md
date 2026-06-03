# Agentic Multimodal RAG Pipeline

A robust, agentic Multimodal Retrieval-Augmented Generation (RAG) system built to parse, index, and query complex documents containing text and visual elements.

## Features

- **Multimodal Indexing**: Extracts and indexes text and images from documents (e.g., PDFs).
- **Agentic Workflow**: Uses LangGraph to manage complex reasoning and retrieval flows, including self-correction mechanisms and a grounding loop.
- **Conversational Interface**: Provides a professional, non-blocking Chainlit user interface for real-time interaction.
- **Automated Evaluation**: Integrated with RAGAS for reference-free evaluation metrics (Faithfulness, Answer Relevancy, Context Utilization).
- **Observability & Analytics**: Streamlit-based dashboard (`dashboard.py`) to monitor aggregate performance and historical evaluation trends.
- **Vector Storage**: Utilizes ChromaDB for vector retrieval and a local document store for full-text caching.

## Tech Stack

- **Core**: Python
- **Orchestration**: LangGraph, LangChain
- **UI**: Chainlit (`chatbot_ui.py`), Streamlit (`dashboard.py`)
- **Evaluation**: Ragas (`ragas_eval.py`, `run_batch_evals.py`)
- **Databases**: ChromaDB (Vector Store)

## Project Structure

- `rag_langgraph.py`: Defines the LangGraph architecture and agentic nodes for the query pipeline.
- `chatbot_ui.py`: Chainlit application script for the user interface.
- `dashboard.py`: Streamlit analytics dashboard for viewing RAGAS evaluation results.
- `ragas_eval.py` / `run_batch_evals.py`: Scripts for running evaluations on agent responses.
- `langgraph.json`: Configuration/state for the LangGraph agent.

<img width="1122" height="852" alt="ss66" src="https://github.com/user-attachments/assets/2ab2dc5e-9830-4b3a-acc6-072dd039ce43" />
<img width="1096" height="837" alt="ss99" src="https://github.com/user-attachments/assets/8e541a60-2a54-4eec-8e7a-f46390cd0a1e" />
<img width="1330" height="654" alt="ss77" src="https://github.com/user-attachments/assets/40bf5c33-a539-461f-bff8-137b7edecd89" />

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   conda create -n rag_env python=3.10
   conda activate rag_env
   ```
   *(Or use `venv`)*

3. **Install dependencies:**
   Ensure you have installed all necessary packages (LangChain, LangGraph, Chainlit, Streamlit, Ragas, ChromaDB, etc.).
   ```bash
   # Make sure to pip install the required libraries if you have a requirements.txt
   # pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys (e.g., OpenAI, LangSmith).
   ```env
   OPENAI_API_KEY=your_api_key_here
   # Other relevant keys
   ```

5. **Run the Chatbot UI:**
   ```bash
   chainlit run chatbot_ui.py -w
   ```

6. **Run the Analytics Dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

## Evaluation

Run batch evaluations to assess the quality of the pipeline's responses:
```bash
python run_batch_evals.py
```
Results are stored in the `ragas_results/` directory and can be visualized via the Streamlit dashboard.
