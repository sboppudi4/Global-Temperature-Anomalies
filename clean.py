"""Initial loading and file-hygiene stage for the temperature pipeline."""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(__file__).with_name("global_temp_dirty_v2.csv")


def load_raw_data(path: Path = INPUT_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load text fields, normalize whitespace, and separate footer rows."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = raw.columns.str.strip()
    raw = raw.map(lambda value: value.strip() if isinstance(value, str) else value)

    date_values = raw["Date"]
    value_values = raw["Temperature_Anomaly"]
    footer_mask = (
        date_values.eq("") & value_values.eq("")
    ) | date_values.str.startswith(("END OF DATA", "Source:"), na=False)

    return raw.loc[~footer_mask].reset_index(drop=True), raw.loc[footer_mask].reset_index(drop=True)


def main() -> None:
    data, footer = load_raw_data()
    print(f"Loaded candidate data rows: {len(data)}")
    print(f"Discarded file-hygiene rows: {len(footer)}")
    print(f"Columns: {', '.join(data.columns)}")


if __name__ == "__main__":
    main()