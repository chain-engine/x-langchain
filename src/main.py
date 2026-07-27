# -*- coding: utf-8 -*-
"""
项目主入口

基于 LangChain 的交互式对话入口。
"""

import sys
import warnings

from core import settings, logger
from core.di import lifespan_container


warnings.filterwarnings(
    "ignore",
    message=".*is not JSON serializable.*",
    category=UserWarning,
)


def interactive_chat() -> None:
    """
    交互式对话模式（支持多轮对话）

    从容器获取 Agent 实例进行对话
    """
    with lifespan_container() as container:
        agent = container.agent

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
        logger.info(f"配置信息: DEBUG={settings.DEBUG}")
        logger.info("正在启动应用...")

        interactive_chat()

    except ImportError as e:
        logger.error(f"缺少依赖模块: {e}")
        logger.info("请运行 'uv sync' 安装依赖")
        sys.exit(1)
    except ConnectionError as e:
        logger.error(f"网络连接失败: {e}")
        logger.info("请检查网络连接和 API 端点配置")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n已取消启动")
        sys.exit(0)
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)


def run() -> None:
    """setuptools entry point 入口"""
    main()


if __name__ == "__main__":
    main()
