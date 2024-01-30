import pathlib
import sys
import typing as t

g_logging_configured = False


LogLevelList = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
LogLevelLiteral = t.Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(
    log_level: str, log_file: pathlib.Path | str | None = None, log_file_level: LogLevelLiteral = "DEBUG"
) -> None:
    global g_logging_configured

    if g_logging_configured:
        return

    from loguru import logger

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

    custom_format = "<level>{level.icon}</level> {message}"

    logger.remove()
    logger.add(sys.stderr, format=custom_format, level=log_level)

    if log_file is not None:
        log_file = pathlib.Path(log_file)
        logger.add(log_file, format=custom_format, level=log_file_level)
        logger.info(f"Logging to {log_file}")

    g_logging_configured = True
