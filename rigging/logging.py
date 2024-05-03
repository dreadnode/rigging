"""
We use loguru for logging. This module provides a function to configure the logging settings.
"""

import pathlib
import sys
import typing as t

from loguru import logger

g_configured: bool = False

LogLevelList = ["trace", "debug", "info", "success", "warning", "error", "critical"]
LogLevelLiteral = t.Literal["trace", "debug", "info", "success", "warning", "error", "critical"]


def configure_logging(
    log_level: str,
    log_file: pathlib.Path | None = None,
    log_file_level: LogLevelLiteral = "debug",
) -> None:
    """
    Configures the loguru settings for the rigging module.

    This is optional, and calling `logger.enable("rigging")` will enable the logging
    and you can control the formatting and log levels using the loguru API.

    Args:
        log_level (str): The desired log level. Valid values are 'TRACE', 'DEBUG', 'INFO',
            'SUCCESS', 'WARNING', 'ERROR', and 'CRITICAL'.
        log_file (pathlib.Path | None, optional): The path to the log file. If None, logging
            will only be done to the console. Defaults to None.
        log_file_level (LogLevelLiteral, optional): The log level for the log file. Valid values
            are 'TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', and 'CRITICAL'.
            Defaults to 'debug'.

    Returns:
        None: This function does not return anything.

    """
    global g_configured

    if g_configured:
        return

    logger.enable("rigging")

    logger.level("TRACE", color="<magenta>", icon="[T]")
    logger.level("DEBUG", color="<blue>", icon="[_]")
    logger.level("INFO", color="<cyan>", icon="[=]")
    logger.level("SUCCESS", color="<green>", icon="[+]")
    logger.level("WARNING", color="<yellow>", icon="[-]")
    logger.level("ERROR", color="<red>", icon="[!]")
    logger.level("CRITICAL", color="<RED>", icon="[x]")

    # Default format:
    # "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    # "<level>{level: <8}</level> | "
    # "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",

    custom_format = "<green>{time:HH:mm:ss.SSS}</green> | <level>{level.icon}</level> {message}"

    logger.remove()
    logger.add(sys.stderr, format=custom_format, level=log_level.upper())

    if log_file is not None:
        logger.add(log_file, format=custom_format, level=log_file_level.upper())
        logger.info(f"Logging to {log_file}")

    g_configured = True
