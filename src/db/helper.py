from sqlglot import exp
from typing import Any



def convert(value: Any, copy: bool = False) -> exp.Expression:
    """A wrapper of exp.convert. Convert a python value into an expression object.
    Raises an error if a conversion is not possible.
    Args:
        value: A python object.
        copy: Whether to copy `value` (only applies to Expressions and collections).

    Returns:
        The equivalent expression object.
    """
    if isinstance(value, str):
        value = value.replace(':', '\:')
    return exp.convert(value= value, copy= copy)
