"""Safe mathematical expression evaluation for derived channels."""

import ast
import re

import numpy as np

from pydatview.common import no_unit

_MATH_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "gradient": np.gradient,
    "degrees": np.degrees,
    "radians": np.radians,
    "mean": np.mean,
    "std": np.std,
}
_MATH_CONSTANTS = {"pi": np.pi, "e": np.e}
_MATH_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Compare,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.UAdd,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _column_array(dataframe, column):
    series = dataframe[column]
    try:
        return series.to_numpy(copy=False)
    except TypeError:
        return series.to_numpy()
    except AttributeError:
        return np.asarray(series)


def _resolve_expression_column(dataframe, token):
    token = token.strip()
    columns = [str(column) for column in dataframe.columns]
    exact_matches = [column for column in columns if column == token]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError("Column name is ambiguous: {}".format(token))
    matches = [column for column in columns if no_unit(column).strip() == token]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("Column not found: {}".format(token))
    raise ValueError("Column name is ambiguous: {}".format(token))


def evaluate_math_expression(dataframe, expression):
    expression = expression.strip()
    if not expression:
        raise ValueError("Expression is empty")

    namespace = dict(_MATH_FUNCTIONS)
    namespace.update(_MATH_CONSTANTS)
    columns = [str(column) for column in dataframe.columns]

    identifier_columns = {}
    for column in columns:
        for candidate in (column, no_unit(column).strip()):
            if candidate.isidentifier() and candidate not in namespace:
                identifier_columns.setdefault(candidate, []).append(column)
    for identifier, matches in identifier_columns.items():
        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) == 1:
            namespace[identifier] = _column_array(dataframe, unique_matches[0])

    token_index = 0

    def replace_column(match):
        nonlocal token_index
        column = _resolve_expression_column(dataframe, match.group(1))
        variable = "_column_{}".format(token_index)
        token_index += 1
        namespace[variable] = _column_array(dataframe, column)
        return variable

    prepared = re.sub(r"\{([^{}]+)\}", replace_column, expression)
    for name in tuple(_MATH_FUNCTIONS) + tuple(_MATH_CONSTANTS):
        prepared = re.sub(r"\bnp\.{}\b".format(re.escape(name)), name, prepared)

    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax: {}".format(exc.msg)) from exc

    for node in ast.walk(tree):
        if not isinstance(node, _MATH_AST_NODES):
            raise ValueError("Unsupported expression element: {}".format(type(node).__name__))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _MATH_FUNCTIONS:
                raise ValueError("Unsupported function")
            if node.keywords:
                raise ValueError("Function keyword arguments are not supported")
        if isinstance(node, ast.Name) and node.id not in namespace:
            raise ValueError("Unknown variable or function: {}".format(node.id))
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise ValueError("Only numeric constants are supported")

    with np.errstate(all="ignore"):
        result = eval(compile(tree, "<calculation>", "eval"), {"__builtins__": {}}, namespace)
    result = np.asarray(result)
    if result.ndim == 0:
        result = np.full(len(dataframe), result.item())
    if result.ndim != 1:
        raise ValueError("Result must be a one-dimensional variable")
    if len(result) != len(dataframe):
        raise ValueError(
            "Result has {:,} values; the table has {:,} rows".format(len(result), len(dataframe))
        )
    if result.dtype.kind not in "biuf":
        raise ValueError("Result must contain numeric values")
    return result

