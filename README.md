# Simple MCP Demo

This is a demonstration project of an intelligent assistant using multiple model providers (OpenAI, OpenRouter, Claude, Amazon Bedrock) and Model Context Protocol (MCP). The assistant can answer various questions and use mathematical tools to solve math calculation problems.

## Features

- Supports multiple model providers:
  - OpenAI (GPT-4o-mini)
  - OpenRouter (Google's Gemini-2.0-flash-exp:free)
  - Claude (Claude 3 Haiku via AWS Bedrock)
  - Nova (Amazon Bedrock Nova Lite)
- Model selection via command line arguments
- Automatically identifies math problems and uses MCP tools for calculations
- Directly answers non-mathematical questions
- Beautiful command-line interface display

## Project Structure

- `client.py`: Main program that handles user input, model selection, and calls the selected model API and MCP tools
- `math_server.py`: MCP math server providing addition and multiplication tools
- `requirements.txt`: List of project dependencies
- `.env.example`: Example environment variables file

## Installation Steps

1. Clone the repository

```bash
git clone <repository URL>
cd simple_mcp_demo
```

2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables

```bash
# Copy the example environment variables file
cp .env.example .env

# Edit the .env file and enter your API keys for the models you want to use
# Open the .env file with your favorite editor
# For example:
# OPENAI_API_KEY=sk-your-openai-api-key
# OPENROUTER_API_KEY=sk-or-your-openrouter-api-key
# BAG_API_BASE=your-claude-api-base-url
```

## Usage

1. Run the program with your preferred model

```bash
# Run with default model (Nova)
python client.py

# Run with a specific model
python client.py --model openai      # Use OpenAI GPT-4o-mini
python client.py --model openrouter  # Use Google Gemini via OpenRouter
python client.py --model claude      # Use Claude 3 Haiku via AWS Bedrock
python client.py --model nova        # Use Amazon Bedrock Nova Lite
```

2. Enter questions

- Math problem examples: "2+6x4", "What is 56 multiplied by 9"
- Non-math question examples: "What is the capital of China", "What is the chemical formula for water"

3. View answers

- For math problems, the program will display the tool calling process and the final calculation result
- For non-math questions, the program will directly display the answer

4. Enter "exit" to quit the program

## Notes

- Make sure you have valid API keys for the models you want to use
- Math tools currently only support addition (add) and multiplication (multiply) operations
- The program will automatically determine the question type, but sometimes more explicit phrasing may be needed

## License

[MIT](LICENSE)
