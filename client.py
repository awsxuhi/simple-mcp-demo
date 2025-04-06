# Import required libraries
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

# Create OpenAI model instance using gpt-4o-mini
model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# System instructions
SYSTEM_PROMPT = """You are an intelligent assistant that can answer various questions.

For mathematical calculation problems (such as addition, multiplication, etc.), you should use the available math tools to help with calculations.
Available math tools include:
- add: for addition calculations
- multiply: for multiplication calculations

For all other types of questions (such as general knowledge, geography, etc.), you should answer directly without trying to use tools.

Always provide useful and accurate answers, do not refuse to answer questions.
"""

# Create server parameters
server_params = StdioServerParameters(
    command="python",
    args=["math_server.py"],
)

# Function to determine if a question might be a math problem
def might_be_math_question(question):
    """
    Simple check to determine if a question might be a math problem
    This is just an initial screening, the final decision is up to the agent
    """
    import re
    # Check if it contains numbers and operators
    has_numbers = bool(re.search(r'\d', question))
    has_operators = any(op in question for op in ['+', '-', '*', '/', 'x', '×', '÷', '='])
    has_math_words = any(word in question.lower() for word in ['add', 'subtract', 'multiply', 'divide', 'equals', 'calculate', 'solve', 'sum', 'difference', 'product', 'quotient', 'plus', 'minus', 'times', 'divided by'])
    
    return has_numbers and (has_operators or has_math_words)

async def run_agent(question):
    """
    Run agent and process user questions
    Let the agent decide whether to use MCP tools
    """
    # First determine if it might be a math question
    if might_be_math_question(question):
        # Might be a math question, connect to MCP server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize connection
                await session.initialize()
                
                # Get tools
                tools = await load_mcp_tools(session)
                
                # Create and run agent, add system instructions
                agent = create_react_agent(model, tools)
                agent_response = await agent.ainvoke({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question}
                    ]
                })
                
                # Check if tools were used
                used_tools = False
                for message in agent_response.get("messages", []):
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        used_tools = True
                        break
                
                return agent_response, used_tools
    else:
        # Non-math question, use model to answer directly
        direct_response = await model.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ])
        
        # Construct response format similar to when using tools
        agent_response = {"messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": direct_response.content}
        ]}
        
        return agent_response, False

def format_agent_response(response, used_tools):
    """Format agent response, determine display content based on whether tools were used"""
    console = Console()
    
    messages = response.get("messages", [])
    
    # Extract AI's final answer
    final_answer = None
    
    # First, find the last assistant message
    for msg in reversed(messages):
        if hasattr(msg, "role") and msg.role == "assistant" and hasattr(msg, "content") and msg.content:
            final_answer = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            final_answer = msg.get("content")
            break
    
    # If no assistant message is found, look for the last non-tool message
    if not final_answer:
        for msg in reversed(messages):
            if (hasattr(msg, "content") and 
                isinstance(msg.content, str) and 
                msg.content and 
                not hasattr(msg, "name") and  # Not a tool message
                not (hasattr(msg, "role") and msg.role == "system")):  # Not a system message
                final_answer = msg.content
                break
            elif (isinstance(msg, dict) and 
                  "content" in msg and 
                  isinstance(msg.get("content"), str) and 
                  msg.get("content") and 
                  "name" not in msg and  # Not a tool message
                  not (msg.get("role") == "system")):  # Not a system message
                final_answer = msg.get("content")
                break
    
    # If tools were used, try to build an answer from the tool results
    if used_tools and not final_answer:
        tool_results = []
        for i, msg in enumerate(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    
                    # Find the corresponding tool result
                    for j in range(i+1, len(messages)):
                        if (hasattr(messages[j], "name") and 
                            messages[j].name == tool_name and 
                            hasattr(messages[j], "content")):
                            tool_results.append({
                                "name": tool_name,
                                "args": tool_args,
                                "result": messages[j].content
                            })
                            break
        
        # If it's a math calculation problem, try to build an answer
        if tool_results and any(t["name"] in ["add", "multiply"] for t in tool_results):
            # Find the final result
            final_result = None
            for tool in reversed(tool_results):  # Search from back to front, take the last result
                if tool["name"] == "add":
                    final_result = tool["result"]
                    break
            
            if not final_result:
                for tool in tool_results:
                    if tool["name"] == "multiply":
                        final_result = tool["result"]
                        break
            
            if final_result:
                final_answer = f"The calculation result is {final_result}."
    
    # If still no answer is found, use default message
    if not final_answer:
        # For non-math questions, use the model to answer directly
        if not used_tools:
            # Extract user question
            user_question = None
            for msg in messages:
                if hasattr(msg, "role") and msg.role == "user" and hasattr(msg, "content"):
                    user_question = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                    user_question = msg.get("content")
                    break
            
            if user_question:
                direct_response = asyncio.run(model.ainvoke([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_question}
                ]))
                final_answer = direct_response.content
            else:
                final_answer = "I cannot understand your question, please ask again."
        else:
            final_answer = "I have processed your request, but cannot generate a final answer."
    
    # If no tools were used, only return the final answer panel
    if not used_tools:
        simple_table = Table(box=box.ROUNDED, expand=True)
        simple_table.add_column("AI Answer", style="cyan")
        simple_table.add_row(Text(final_answer, style="bold white"))
        return simple_table
    
    # If tools were used, return the complete interaction process
    main_table = Table(box=box.ROUNDED, expand=True)
    main_table.add_column("Interaction Process", style="cyan")
    
    # Extract user question
    user_question = None
    for msg in messages:
        if hasattr(msg, "role") and msg.role == "user" and hasattr(msg, "content"):
            user_question = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
            user_question = msg.get("content")
            break
        elif hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            user_question = msg.content
            break
    
    if user_question:
        question_panel = Panel(
            Text(f"{user_question}", style="bold green"),
            title="User Question",
            border_style="green"
        )
        main_table.add_row(question_panel)
    
    # Extract tool calls and results
    tool_calls = []
    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                # Find the corresponding tool result
                tool_result = None
                for j in range(i+1, len(messages)):
                    if (hasattr(messages[j], "name") and 
                        messages[j].name == tool_name and 
                        hasattr(messages[j], "content")):
                        tool_result = messages[j].content
                        break
                
                tool_calls.append({
                    "name": tool_name,
                    "args": tool_args,
                    "result": tool_result
                })
    
    if tool_calls:
        tools_table = Table(show_header=True, box=box.SIMPLE)
        tools_table.add_column("Tool", style="magenta")
        tools_table.add_column("Parameters", style="yellow")
        tools_table.add_column("Result", style="cyan")
        
        for tool in tool_calls:
            tools_table.add_row(
                tool["name"],
                str(tool["args"]),
                str(tool["result"])
            )
        
        tools_panel = Panel(
            tools_table,
            title="Tool Calls",
            border_style="blue"
        )
        main_table.add_row(tools_panel)
    
    # Add final answer
    answer_panel = Panel(
        Text(final_answer, style="bold white"),
        title="AI Answer",
        border_style="cyan"
    )
    main_table.add_row(answer_panel)
    
    return main_table

# Main function
if __name__ == "__main__":
    console = Console()
    console.print("\n")
    console.print(Panel.fit("MCP Intelligent Assistant Demo", style="bold magenta"))
    console.print("\n")
    
    while True:
        try:
            # Get user input
            user_question = input("Enter your question (type 'exit' to quit): ")
            if user_question.lower() == 'exit':
                console.print("[yellow]Thank you for using the assistant. Goodbye![/yellow]\n")
                break
                
            if not user_question.strip():
                console.print("[red]Question cannot be empty, please try again[/red]\n")
                continue
            
            # Run agent
            console.print("[cyan]Thinking...[/cyan]")
            result, used_tools = asyncio.run(run_agent(user_question))
            
            # Use Rich to format output
            formatted_result = format_agent_response(result, used_tools)
            console.print(formatted_result)
            console.print("\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Program interrupted, thank you for using![/yellow]\n")
            break
        except Exception as e:
            console.print(f"[red]An error occurred: {str(e)}[/red]\n")
