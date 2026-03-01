def linearize_table(table, max_rows=6):
    rows = []
    for i, row in enumerate(table[:max_rows]):
        cells = [f"c{j}={cell}" for j, cell in enumerate(row)]
        rows.append(f"row{i}: " + " | ".join(cells))
    return " ; ".join(rows)