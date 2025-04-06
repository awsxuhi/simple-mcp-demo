When the client.py has the following as its content, the output will be as follows:

```python
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-4o")

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["math_server.py"],
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            return agent_response

# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result)
```

output of `python client.py`:

```shell
 simple_mcp_demo python client.py
[04/06/25 12:25:04] INFO     Processing request of type ListToolsRequest                                                                                                                                                                                                                       server.py:534
[04/06/25 12:25:05] INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                        server.py:534
                    INFO     Processing request of type CallToolRequest                                                                                                                                                                                                                        server.py:534
{'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='6ff1b28f-890e-40c7-bc07-57ddc423de0c'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_eNGJ4sbf5Kl7C1h6RNSUclce', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function'}, {'id': 'call_1qxqhr8c0dy8FSXclc5yYBqA', 'function': {'arguments': '{"a": 8, "b": 12}', 'name': 'multiply'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 51, 'prompt_tokens': 77, 'total_tokens': 128, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_6dd05565ef', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-8c063485-f242-4acf-84d4-9b736f3f104a-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_eNGJ4sbf5Kl7C1h6RNSUclce', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 8, 'b': 12}, 'id': 'call_1qxqhr8c0dy8FSXclc5yYBqA', 'type': 'tool_call'}], usage_metadata={'input_tokens': 77, 'output_tokens': 51, 'total_tokens': 128, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='8', name='add', id='97e9101f-f287-411b-bb99-3bfba7bac26a', tool_call_id='call_eNGJ4sbf5Kl7C1h6RNSUclce'), ToolMessage(content='96', name='multiply', id='83058942-a482-4674-a7dc-b9d885472f4b', tool_call_id='call_1qxqhr8c0dy8FSXclc5yYBqA'), AIMessage(content='The result of \\((3 + 5) \\times 12\\) is 96.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 143, 'total_tokens': 165, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_6dd05565ef', 'finish_reason': 'stop', 'logprobs': None}, id='run-dc317b73-079b-4230-bac0-26c8c10efb9f-0', usage_metadata={'input_tokens': 143, 'output_tokens': 22, 'total_tokens': 165, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}
sample-agent-py3.12➜  simple_mcp_demo 

```