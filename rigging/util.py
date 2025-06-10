"""
Common utilities used throughout the library.
"""

import asyncio
import functools
import inspect
import re
import types
import typing as t
from json import JSONDecoder
from threading import Thread

import jsonref  # type: ignore [import-untyped]
from pydantic import alias_generators

R = t.TypeVar("R")

# Async utilities
#
# TODO: Should this be a global? Is that safe with multiple threads?
# I originally looked as TLS for this, but I wasn't confident in the
# complexity. I'd also imagine this is a common pattern with some
# best practices available.

g_event_loop: asyncio.AbstractEventLoop | None = None


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _get_event_loop() -> asyncio.AbstractEventLoop:
    global g_event_loop  # noqa: PLW0603

    if g_event_loop is None:
        g_event_loop = asyncio.new_event_loop()
        thread = Thread(target=_run_loop, args=(g_event_loop,), daemon=True)
        thread.start()

    return g_event_loop


@t.overload
def await_(coros: t.Coroutine[t.Any, t.Any, R]) -> R: ...


@t.overload
def await_(*coros: t.Coroutine[t.Any, t.Any, R]) -> list[R]: ...


def await_(*coros: t.Coroutine[t.Any, t.Any, R]) -> R | list[R]:  # type: ignore [misc]
    """
    A utility function that allows awaiting coroutines in a managed thread.

    Args:
        *coros: Variable number of coroutines to await.

    Returns:
        A single result if one coroutine is passed or a list of results if multiple coroutines are passed.
    """
    loop = _get_event_loop()
    tasks = [asyncio.run_coroutine_threadsafe(coro, loop) for coro in coros]
    results = [task.result() for task in tasks]
    if len(coros) == 1:
        return results[0]
    return results


# JSON


def deref_json(obj: dict[str, t.Any], *, is_json_schema: bool = False) -> dict[str, t.Any]:
    """
    Light wrapper around jsonref.replace_refs() to dereference JSON objects which might
    contain JSON Schema references ($ref).

    Args:
        obj: JSON object to dereference.
        is_json_schema: See jsonref.replace_refs() for details on this parameter.

    Returns:
        A new JSON object with all references resolved.
    """
    return jsonref.replace_refs(  # type: ignore [no-any-return]
        obj,
        jsonschema=is_json_schema,
        proxies=False,
        lazy_load=False,
    )


def extract_json_objects(text: str) -> list[tuple[dict[str, t.Any], slice]]:
    """
    Find JSON objects in text using JSONDecoder.raw_decode().

    Does not attempt to look for JSON arrays, text, or other JSON types outside
    of a parent JSON object.

    Args:
        text: Text to search for JSON objects

    Returns:
        A list of tuples containing (JSON object, slice) where slice indicates
        the position in the original text where the object was found.
    """
    decoder = JSONDecoder()
    results = []
    pos = 0

    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            # Create slice representing the position in the original text
            json_slice = slice(match, match + index)
            results.append((result, json_slice))
            pos = match + index
        except ValueError:
            pos = match + 1

    return results


# XML Formatting


def unescape_xml(xml_string: str) -> str:
    """
    Unescape XML special characters in a string.
    """
    unescaped = re.sub(r"&amp;", "&", xml_string)
    unescaped = re.sub(r"&lt;", "<", unescaped)
    unescaped = re.sub(r"&gt;", ">", unescaped)
    unescaped = re.sub(r"&apos;", "'", unescaped)
    return re.sub(r"&quot;", '"', unescaped)


def escape_xml(xml_string: str) -> str:
    """
    Escape XML special characters in a string.
    """
    escaped = xml_string.replace(r"&", "&amp;")
    escaped = escaped.replace(r"<", "&lt;")
    escaped = escaped.replace(r">", "&gt;")
    escaped = escaped.replace(r"'", "&apos;")
    return escaped.replace(r'"', "&quot;")


def unescape_cdata_tags(xml_string: str) -> str:
    """
    Unescape double-escaped CDATA tags in an XML string.
    """

    def unescape_cdata(match: re.Match[str]) -> str:
        return unescape_xml(match.group(1))

    return re.sub(
        r"&lt;!\[CDATA\[(.*?)\]\]&gt;",  # The CDATA itself is escaped at this point,
        unescape_cdata,
        xml_string,
        flags=re.DOTALL,
    )


def to_snake(text: str) -> str:
    """
    Convert a string to snake_case.
    """
    return alias_generators.to_snake(text).replace("-", "_")


def to_xml_tag(text: str) -> str:
    """
    Convert a string to a valid XML tag name.
    """
    return to_snake(text).replace("_", "-").strip("-")


# Name resolution


def get_qualified_name(obj: t.Callable[..., t.Any]) -> str:
    """
    Return a best guess at the qualified name of a callable object.
    This includes functions, methods, and callable classes.
    """
    if obj is None or not callable(obj):
        return "unknown"

    module = inspect.getmodule(obj)
    module_name = module.__name__ if module else ""

    # Partial functions
    if isinstance(obj, functools.partial):
        base_name = get_qualified_name(obj.func)
        return f"partial({base_name})"

    # Methods
    if isinstance(obj, types.MethodType):
        class_name = obj.__self__.__class__.__name__
        method_name = obj.__func__.__name__
        return f"{class_name}.{method_name}"

    # Functions
    if isinstance(obj, types.FunctionType):
        # Check if it's a wrapped function
        if hasattr(obj, "__wrapped__"):
            original_name = get_qualified_name(obj.__wrapped__)
            return f"wrapped({original_name})"

        name = obj.__qualname__ or obj.__name__
        return f"{module_name}.{name}" if module_name != "__main__" else name

    # Callable classes
    if callable(obj):
        if isinstance(obj, type):
            return obj.__qualname__
        return f"{obj.__class__.__qualname__}.__call__"

    # Fallback
    return obj.__class__.__qualname__


# Formatting


def shorten_string(content: str, max_length: int, *, sep: str = "...") -> str:
    """
    Return a string at most max_length characters long by removing the middle of the string.
    """
    if len(content) <= max_length:
        return content

    remaining = max_length - len(sep)
    if remaining <= 0:
        return sep

    middle = remaining // 2
    return content[:middle] + sep + content[-middle:]


def truncate_string(content: str, max_length: int, *, suf: str = "...") -> str:
    """
    Return a string at most max_length characters long by removing the end of the string.
    """
    if len(content) <= max_length:
        return content

    remaining = max_length - len(suf)
    if remaining <= 0:
        return suf

    return content[:remaining] + suf


# List utilities


def flatten_list(nested_list: t.Iterable[t.Iterable[t.Any] | t.Any]) -> list[t.Any]:
    """
    Recursively flatten a nested list into a single list.
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


# Audio

AudioFormat = t.Literal["wav", "mp3", "ogg", "flac"]


def identify_audio_format(data: bytes) -> AudioFormat | None:
    """
    Identify audio format by checking the first few bytes of data
    """
    if len(data) < 12:  # noqa: PLR2004
        return None  # Not enough data to identify format

    header = data[:12]

    signatures: dict[bytes, AudioFormat] = {
        b"RIFF": "wav",  # WAV files start with 'RIFF'
        b"ID3": "mp3",  # MP3 files often start with 'ID3' (ID3 tag)
        b"\xff\xfb": "mp3",  # MP3 files without ID3 tag
        b"\xff\xf3": "mp3",  # MP3 files (MPEG-1 Layer 3)
        b"\xff\xf2": "mp3",  # MP3 files (MPEG-2 Layer 3)
        b"OggS": "ogg",  # Ogg files
        b"fLaC": "flac",  # FLAC files
    }

    for signature, format_name in signatures.items():
        if header.startswith(signature):
            return format_name

    # Check for MP3 without ID3 tag (check for MP3 frame sync)
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:  # noqa: PLR2004
        return "mp3"

    return None
