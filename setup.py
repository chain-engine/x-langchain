from setuptools import setup

setup(
    name="x-langchain",
    version="0.1.0",
    description="LangChain Learning and Practice Project",
    readme="README.md",
    requires_python=">=3.11",
    py_modules=["main"],
    install_requires=[
        # Core LangChain
        "langchain>=0.2.0",
        "langchain-openai>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "langchain-community>=0.4.1",
        # Configuration
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        # LLM Providers
        "dashscope>=1.25.11",
        # Tools
        "duckduckgo-search>=8.1.1",
        "ddgs>=9.11.1",
        "sqlalchemy>=2.0.0",
        "pymysql>=1.1.0",
        "langchain-mcp-adapters>=0.2.2",
        # Utilities
        "loguru>=0.7.3",
    ],
)
