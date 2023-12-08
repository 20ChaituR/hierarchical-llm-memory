# Hierarchical LLM Memory

Allows an LLM to explore your filesystem and read code files. The memory system is hierarchical, so even long code files are perfectly fine (as long as they're indented well).

## Setup

Clone the directory:

```
git clone https://github.com/20ChaituR/hierarchical-llm-memory.git
cd hierarchical-llm-memory
```

Setup the environment, installing all the necessary libraries:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Follow the instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key) to get an OpenAI API key. Add the key to your environment:

```
export OPENAI_API_KEY='your-api-key-here'
```

## Running the Code

To run the code, run the following command:

```
python main.py your_directory "your query"
```

Where `your_directory` should be the directory which you want ChatGPT to have access to, and `your query` is the question you want to ask about that directory.
