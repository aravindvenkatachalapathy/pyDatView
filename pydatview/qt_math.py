"""Safe mathematical expression evaluation for derived channels."""

import ast
import os
import re

import numpy as np
import pandas as pd

from pydatview.common import no_unit


def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    window = int(window)

    if window < 1:
        raise ValueError("Moving-average window must be >= 1")

    if window > len(x):
        raise ValueError(
            "Moving-average window ({}) is larger than data length ({})".format(
                window, len(x)
            )
        )

    kernel = np.ones(window, dtype=float) / window

    return np.convolve(x, kernel, mode="same")

def root_mean_square(x, window):
    x = np.asarray(x, dtype=float)
    window = int(window)

    if window < 1:
        raise ValueError("RMS window must be >= 1")

    if window > len(x):
        raise ValueError(
            "RMS window ({}) is larger than data length ({})".format(
                window, len(x)
            )
        )

    kernel = np.ones(window, dtype=float) / window

    return np.sqrt(np.convolve(x**2, kernel, mode="same"))


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
    "moving_average": moving_average,
    "root_mean_square": root_mean_square,
}
_MATH_CONSTANTS = {"pi": np.pi, "e": np.e}


def _time_column(dataframe):
    candidates = [
        str(column) for column in dataframe.columns
        if str(column).lower() != "index"
    ]
    for column in candidates:
        if no_unit(column).strip().lower() in ("time", "t"):
            return column
    for column in candidates:
        if "time" in no_unit(column).strip().lower():
            return column
    raise ValueError("No time column found; specify one with x={Column name}")


def trim_rows(dataframe, x=None, start=None, stop=None):
    if start is None and stop is None:
        raise ValueError("trim requires start, stop, or both")
    column = _time_column(dataframe) if x is None else _resolve_expression_column(
        dataframe,
        str(x),
    )
    values = dataframe[column]
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        start = pd.to_datetime(start) if start is not None else None
        stop = pd.to_datetime(stop) if stop is not None else None
    mask = np.ones(len(dataframe), dtype=bool)
    if start is not None:
        mask &= np.asarray(values >= start)
    if stop is not None:
        mask &= np.asarray(values <= stop)
    return mask


_TABLE_TRANSFORMS = {
    "trim": trim_rows,
}
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


def _transform_argument(node, column_bindings):
    if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, str, bool, type(None))):
        return node.value
    if isinstance(node, ast.Name) and node.id in column_bindings:
        return column_bindings[node.id]
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return (
            node.operand.value
            if isinstance(node.op, ast.UAdd)
            else -node.operand.value
        )
    raise ValueError("Transform arguments must be literals or {column names}")


def evaluate_table_script(dataframe, script):
    """Apply a safe sequence of registered row transforms to a table."""
    lines = [
        line.strip() for line in script.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("Transform script is empty")

    current = dataframe
    row_positions = np.arange(len(dataframe))
    for line_number, line in enumerate(lines, start=1):
        column_bindings = {}

        def replace_column(match):
            column = _resolve_expression_column(current, match.group(1))
            token = "_transform_column_{}".format(len(column_bindings))
            column_bindings[token] = column
            return token

        prepared = re.sub(r"\{([^{}]+)\}", replace_column, line)
        try:
            tree = ast.parse(prepared, mode="eval")
        except SyntaxError as exc:
            raise ValueError(
                "Invalid transform on line {}: {}".format(
                    line_number,
                    exc.msg,
                )
            ) from exc
        call = tree.body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError(
                "Line {} must contain one transform call".format(line_number)
            )
        transform = _TABLE_TRANSFORMS.get(call.func.id)
        if transform is None:
            raise ValueError("Unsupported table transform: {}".format(call.func.id))
        if call.args:
            raise ValueError("Use named arguments for table transforms")
        arguments = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise ValueError("Expanded keyword arguments are not supported")
            arguments[keyword.arg] = _transform_argument(
                keyword.value,
                column_bindings,
            )

        mask = np.asarray(transform(current, **arguments))
        if mask.dtype.kind != "b" or mask.ndim != 1 or len(mask) != len(current):
            raise ValueError(
                "Transform {} did not return one row decision per value".format(
                    call.func.id
                )
            )
        current = current.loc[mask].copy()
        row_positions = row_positions[mask]
        if len(current) == 0:
            raise ValueError(
                "Transform {} removed every row".format(call.func.id)
            )

    current.reset_index(drop=True, inplace=True)
    if len(current.columns) and str(current.columns[0]).lower() == "index":
        current.iloc[:, 0] = np.arange(len(current))
    return current, row_positions


def transform_file_tables(tab_list, table_index, suffix, script):
    """Create transformed copies for every table in the selected file."""
    selected = tab_list[table_index]
    selected_group = selected.source_metadata.get('transform_group')
    if selected_group is not None:
        target_indices = [
            index for index, table in enumerate(tab_list)
            if table.source_metadata.get('transform_group') == selected_group
        ]
    elif selected.filename:
        normalized = os.path.normcase(os.path.abspath(selected.filename))
        target_indices = [
            index for index, table in enumerate(tab_list)
            if table.filename
            and os.path.normcase(os.path.abspath(table.filename)) == normalized
            and table.source_metadata.get('transform_group') is None
        ]
    else:
        target_indices = [table_index]

    existing_names = {table.name for table in tab_list}
    pending = []
    trimmed_count = 0
    static_count = 0
    for index in target_indices:
        source = tab_list[index]
        output_name = source.nickname + suffix
        try:
            _time_column(source.data)
        except ValueError:
            transformed_data = source.data.copy(deep=False)
            row_positions = np.arange(len(source.data))
            static_count += 1
        else:
            transformed_data, row_positions = evaluate_table_script(
                source.data,
                script,
            )
            trimmed_count += 1
        transformed = source.transformed(
            transformed_data,
            output_name,
            row_positions=row_positions,
        )
        if transformed.name in existing_names:
            raise ValueError(
                "A table named '{}' already exists".format(output_name)
            )
        pending.append(transformed)
        existing_names.add(transformed.name)

    if trimmed_count == 0:
        raise ValueError(
            "No time-dependent tables were found in the selected file"
        )
    return pending, target_indices, trimmed_count, static_count
