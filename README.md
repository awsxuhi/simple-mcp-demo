# Simple MCP Demo

This is a demonstration project of an intelligent assistant using OpenAI API and Model Context Protocol (MCP). The assistant can answer various questions and use mathematical tools to solve math calculation problems.

## Features

- Uses OpenAI's GPT-4o-mini model
- Automatically identifies math problems and uses MCP tools for calculations
- Directly answers non-mathematical questions
- Beautiful command-line interface display

## Project Structure

- `client.py`: Main program that handles user input and calls OpenAI API and MCP tools
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

# Edit the .env file and enter your OpenAI API key
# Open the .env file with your favorite editor
# For example:
# OPENAI_API_KEY=sk-your-api-key
```

## Usage

1. Run the program

```bash
python client.py
```

2. Enter questions

- Math problem examples: "2+6x4", "What is 56 multiplied by 9"
- Non-math question examples: "What is the capital of China", "What is the chemical formula for water"

3. View answers

- For math problems, the program will display the tool calling process and the final calculation result
- For non-math questions, the program will directly display the answer

4. Enter "exit" to quit the program

## Notes

- Make sure you have a valid OpenAI API key
- Math tools currently only support addition (add) and multiplication (multiply) operations
- The program will automatically determine the question type, but sometimes more explicit phrasing may be needed

## License

[MIT](LICENSE)
