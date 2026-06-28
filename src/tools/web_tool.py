from typing import Any, Dict, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="搜索查询词")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "检索互联网信息"
    args_schema: type[WebSearchArgs] = WebSearchArgs

    def _run(self, query: str) -> str:
        """检索互联网信息

        Args:
            query: 搜索查询词

        Returns:
            互联网搜索结果字符串
        """
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results: List[Dict[str, Any]] = list(ddgs.text(query, max_results=5))

            if not results:
                return f"未找到关于 '{query}' 的搜索结果"

            # 格式化搜索结果
            formatted_results: List[str] = []
            for i, result in enumerate(results, 1):
                title: str = result.get("title", "无标题")
                href: str = result.get("href", "")
                body: str = result.get("body", "无摘要")
                formatted_results.append(
                    f"{i}. {title}\n   链接: {href}\n   摘要: {body}\n"
                )

            return f"关于 '{query}' 的搜索结果:\n\n" + "\n".join(formatted_results)

        except ImportError:
            # 如果 ddgs 未安装，使用简单的模拟实现
            return f"搜索结果：{query}\n\n注意：要获取真实的网络搜索结果，请安装 ddgs 库：\n  uv add ddgs"
        except Exception as e:
            return f"搜索时发生错误: {str(e)}"


if __name__ == "__main__":
    tool: WebSearchTool = WebSearchTool()
    print(tool._run("Python programming"))
