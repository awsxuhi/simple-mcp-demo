# math_server.py
from mcp.server.fastmcp import FastMCP
import ast
import operator

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# 暂时注释掉evaluate工具，以测试链式调用add和multiply工具
# @mcp.tool()
# def evaluate(expression: str) -> int:
#     """Evaluate a mathematical expression like '3+4*5'"""
#     try:
#         # 使用Python的ast模块安全地计算表达式
#         # 这比eval更安全，因为它只允许数学运算
#         allowed_operators = {
#             ast.Add: operator.add,
#             ast.Sub: operator.sub,
#             ast.Mult: operator.mul,
#             ast.Div: operator.truediv,
#             ast.USub: operator.neg,
#         }
#         
#         def eval_expr(node):
#             if isinstance(node, ast.Num):
#                 return node.n
#             elif isinstance(node, ast.BinOp):
#                 return allowed_operators[type(node.op)](
#                     eval_expr(node.left),
#                     eval_expr(node.right)
#                 )
#             elif isinstance(node, ast.UnaryOp):
#                 return allowed_operators[type(node.op)](eval_expr(node.operand))
#             else:
#                 raise TypeError(f"Unsupported type: {node}")
#         
#         # 解析表达式
#         parsed_expr = ast.parse(expression, mode='eval').body
#         result = eval_expr(parsed_expr)
#         return result
#     except Exception as e:
#         return f"Error evaluating expression: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
