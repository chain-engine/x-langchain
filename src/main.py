# -*- coding: utf-8 -*-
"""
项目主入口

基于新架构的交互式对话入口。
整合 LLM、Memory、Planning、Action、Tools 五大核心子系统。
"""

import sys
import os
import warnings

from .core import settings, logger
from .agent import Agent, AgentConfig
from .memories import ConversationHistoryMemory


warnings.filterwarnings(
    "ignore",
    message=".*is not JSON serializable.*",
    category=UserWarning,
)


def interactive_chat(agent: Agent) -> None:
    """
    交互式对话模式（支持多轮对话）

    Args:
        agent: Agent 实例
    """
    print("\n" + "=" * 50)
    print("欢迎使用智能助手！")
    print("输入 'exit'、'quit' 或 '退出' 结束对话")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("你: ").strip()

            if user_input.lower() in ["exit", "quit", "退出", "离开", "bye", "再见"]:
                print("\n感谢使用，再见！")
                break

            if not user_input:
                continue

            print()
            # 调用 Agent
            response = agent.invoke(user_input)

            if response.success:
                print(response.content)
            else:
                print(f"抱歉，{response.content}")

            if response.tool_results:
                print(f"\n[调用了 {len(response.tool_results)} 个工具，迭代 {response.iterations} 次]")

            print()

        except KeyboardInterrupt:
            print("\n\n检测到中断信号，正在退出...")
            break
        except EOFError:
            print("\n\n输入结束，正在退出...")
            break
        except Exception as e:
            logger.error(f"处理输入时出错: {e}")
            print("抱歉，处理您的请求时出现了错误，请重试。\n")


def main() -> None:
    """主函数"""
    try:
        model_name = os.getenv("MODEL_NAME", "deepseek")

        logger.info(f"正在验证模型配置: {model_name}")
        if not settings.validate_model_config(model_name):
            logger.error(f"{model_name} 模型的配置不完整，请检查 .env 文件中的配置")
            sys.exit(1)

        logger.info(f"配置信息: DEBUG={settings.DEBUG}, STRUCTURED={settings.STRUCTURED}")
        logger.info("正在创建 Agent...")

        # 创建配置
        config = AgentConfig(
            model_provider=model_name,
            system_prompt="""你是一个智能助手，可以帮助用户完成各种任务。

当用户需要实时信息或外部数据时，优先调用工具。使用工具后，请清晰总结工具结果，
不要编造事实。遇到数据库问题时，请遵循 TextToSQL 流程：改写问题、查看表结构、
生成 SQL、校验 SQL、执行 SQL，然后用自然语言解释结果。""",
            enable_memory=True,
            enable_tools=True,
            debug=settings.DEBUG,
        )

        # 创建记忆
        memory = ConversationHistoryMemory(
            max_messages=100,
            system_message=config.system_prompt,
        )

        # 创建 Agent
        agent = Agent(
            config=config,
            memory=memory,
        )

        # 启动交互式对话
        interactive_chat(agent)

    except ImportError as e:
        logger.error(f"缺少依赖模块: {e}")
        logger.info("请运行 'uv sync' 安装依赖")
        sys.exit(1)
    except ConnectionError as e:
        logger.error(f"网络连接失败: {e}")
        logger.info("请检查网络连接和 API 端点配置")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"参数错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"错误: {e}")
        if settings.DEBUG:
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
