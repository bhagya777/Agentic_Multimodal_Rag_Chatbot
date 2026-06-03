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
<br><br>

<img width="1096" height="837" alt="ss99" src="https://github.com/user-attachments/assets/8e541a60-2a54-4eec-8e7a-f46390cd0a1e" />
<br><br>

<img width="1330" height="654" alt="ss77" src="https://github.com/user-attachments/assets/40bf5c33-a539-461f-bff8-137b7edecd89" />


