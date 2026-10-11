"""Small Markdown table formatter used by optional vault reports."""
import pandas as pd


def markdown_table(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v)
                                       for v in row) + " |")
    return "\n".join(lines)
