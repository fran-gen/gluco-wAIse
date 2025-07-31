# Gluco-wAIse

Gluco-wAIse is a Python assistant that helps people with diabetes make informed food choices.\
It uses [LangGraph](https://www.langgraph.dev/), [LangChain](https://www.langchain.com/), and supports both CLI and UI interfaces.

---

## What It Does

Gluco-wAIse offers two main functionalities:

1. **Conversational Q&A**: Users can ask questions about diabetes-related nutrition. The assistant uses a Retrieval-Augmented Generation (RAG) approach to search a custom-built knowledge base of Q&A pairs and respond accurately.

2. **Visual Food Analysis (UI only)**: In the Chainlit web UI, users can upload images (e.g., of meals). The assistant uses a Vision Language Model (VLM) to analyze the image and evaluate whether the food aligns with diabetic dietary recommendations.

   - PDF files can also be uploaded to dynamically update the knowledge base, which is embedded into a new FAISS vector store. This feature is still in development.
   - Optionally, the assistant can generate Word documents upon request using an integrated tool.

---

## Requirements

- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- Poetry 2.1.3 (recommended)
  - To install this version:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 - --version 2.1.3
    ```

---

## Setup (Conda + Poetry)

### 1. Create and activate a Conda environment

```bash
conda create -n gluco-wise python=3.13.5 
conda activate gluco-wise
```

### 2. Tell Poetry to use the Conda Python interpreter

From the project root (`gluco-wAIse`):

```bash
poetry env use $(which python)
```

### 3. Install project dependencies

```bash
poetry install
```

### 4. Activate the Poetry virtual environment

```bash
source deactivate && source $(poetry env info --path)/bin/activate
```

---

## Environment Variables

Make sure to create a `.env` file inside the root `gluco-wAIse` directory. The file should include the following keys:

```env
OPENAI_API_KEY=
LANGSMITH_API_KEY=
```

---

## Knowledge Base Preparation

Before running the agent, generate the JSON knowledge base file used for retrieval:

```bash
python json_generator.py
```

This will create a file at `drafting-agent/data/kb/diabetes_kb.json` containing questions and answers that the assistant uses to respond to user queries.

To make this data searchable by the assistant, you must also build the FAISS vector store from the knowledge base. This is done by running:

```bash
python gluco_wAIse/vectorstore_generator.py
```

This script loads the JSON file, converts questions into vector embeddings using OpenAI, and saves them to a FAISS index at `gluco-wAIse/vectorstore/food_kb_index`.

---

## Usage

### Option 1: Run from the terminal (interactive CLI)

```bash
cd gluco_wAIse
poetry run run-agent "<your message>"
```

You will enter an interactive mode.\
By typing your question (e.g., "What are good snacks for type 2 diabetes?") you will receive a response.

---

### Option 2: Run with Chainlit (web-based UI)

If you prefer a user interface, launch the assistant with Chainlit from the project root:

```bash
chainlit run gluco_wAIse/chainlit_app.py
```

This will start a local server (usually at [http://localhost:8000](http://localhost:8000)) where you can interact with the assistant in your browser.

---

## LangGraph CLI (Optional)

To install the LangGraph CLI for local development:

```bash
pip install --upgrade "langgraph-cli[inmem]"
```

Then run:

```bash
cd gluco_wAIse
langgraph dev
```

This launches a local development server for testing and visualizing your LangGraph application.\
You will also be able to see the structure of the agent's graph and interact with it through a visual UI. Make sure `.env` is within the `gluco_wAIse` directory.

More information: [https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/)


