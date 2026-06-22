"""
日志配置工具
------------
统一的项目日志配置，支持控制台和文件双输出。
日志级别和文件路径从 config.settings 读取。
"""

import logging
import sys
from typing import Optional
from config.settings import settings


def setup_logger(
    name: str = "travel_scenic_rag",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    创建并配置项目日志器。

    参数：
        name: 日志器名称
        level: 日志级别（默认从 settings 读取）
        log_file: 日志文件路径（为 None 时仅控制台输出）

    返回：
        配置完成的 logging.Logger 实例
    """
    # 使用传入参数或 fallback 到全局配置
    log_level: str = level or settings.LOG_LEVEL
    log_file_path: Optional[str] = log_file or settings.LOG_FILE

    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件 handler（可选）
    if log_file_path:
        try:
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            # 文件 handler 创建失败不应阻塞程序启动
            print(f"[WARNING] 日志文件 {log_file_path} 创建失败: {e}")

    return logger
