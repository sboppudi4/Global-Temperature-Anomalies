"""Initial loading and file-hygiene stage for the temperature pipeline."""

from pathlib import Path
import re
from datetime import datetime
from typing import Optional

import pandas as pd


INPUT_PATH = Path(__file__).with_name("global_temp_dirty_v2.csv")
OUTPUT_PATH = Path(__file__).with_name("cleaned_monthly.csv")
MISSING_TOKENS = {"", ".", "--", "NaN", "null", "NA", "N/A", "#N/A", "n/a", "missing"}
MALFUNCTION_CODES = {"500", "-500", "999", "-999"}


def _expand_two_digit_year(year: str) -> int:
    year_number = int(year)
    return 1900 + year_number if year_number >= 26 else 2000 + year_number


def _parse_month_date(value: str) -> Optional[pd.Timestamp]:
    value = value.strip()
    if value in MISSING_TOKENS:
        return None

    patterns = (
        (r"^(\d{4})(\d{2})$", lambda match: (int(match[1]), int(match[2]))),
        (r"^(\d{4})[-/.](\d{1,2})(?:[-/.]\d{1,2})?$", lambda match: (int(match[1]), int(match[2]))),
        (r"^(\d{1,2})[-/](\d{4})$", lambda match: (int(match[2]), int(match[1]))),
        (r"^(\d{1,2})[-/](\d{2})$", lambda match: (_expand_two_digit_year(match[2]), int(match[1]))),
        (r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$", lambda match: (
            _expand_two_digit_year(match[3]) if len(match[3]) == 2 else int(match[3]),
            int(match[1]),
        )),
        (r"^([A-Za-z]+)[ -](\d{2}|\d{4})$", lambda match: (
            _expand_two_digit_year(match[2]) if len(match[2]) == 2 else int(match[2]),
            datetime.strptime(match[1], "%B").month if len(match[1]) > 3 else datetime.strptime(match[1], "%b").month,
        )),
        (r"^(\d{4})\s+([A-Za-z]+)$", lambda match: (
            int(match[1]),
            datetime.strptime(match[2], "%B").month if len(match[2]) > 3 else datetime.strptime(match[2], "%b").month,
        )),
    )

    for pattern, parts in patterns:
        match = re.fullmatch(pattern, value)
        if match:
            try:
                year, month = parts(match)
                return pd.Timestamp(year=year, month=month, day=1)
            except ValueError:
                return None
    return None


def _looks_like_anomaly(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:[.,]\d+)?|\d+\.\d+°C)", value.strip()))


def _looks_like_date(value: str) -> bool:
    return _parse_month_date(value) is not None


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


def standardize_dates(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Repair swapped date/value fields and parse dates to month timestamps."""
    data = data.copy()
    swapped_count = 0
    unparsed_rows = []

    for index in data.index:
        date_value = data.at[index, "Date"]
        anomaly_value = data.at[index, "Temperature_Anomaly"]
        if (
            (_looks_like_anomaly(date_value) or date_value in MISSING_TOKENS)
            and _looks_like_date(anomaly_value)
        ):
            data.at[index, "Date"], data.at[index, "Temperature_Anomaly"] = anomaly_value, date_value
            swapped_count += 1

        parsed_date = _parse_month_date(data.at[index, "Date"])
        if parsed_date is None:
            unparsed_rows.append(data.loc[index].to_dict())
        else:
            data.at[index, "Date"] = parsed_date

    data["Date"] = pd.to_datetime(data["Date"])
    return data, pd.DataFrame(unparsed_rows), swapped_count


def parse_anomalies(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Convert anomaly text to floats and classify missing or malfunction values."""
    data = data.copy()
    invalid_rows = []
    malfunction_count = 0
    parsed_values = []

    for index, raw_value in data["Temperature_Anomaly"].items():
        normalized = raw_value.strip()
        if normalized in MISSING_TOKENS:
            parsed_values.append(float("nan"))
            continue
        if normalized in MALFUNCTION_CODES:
            parsed_values.append(float("nan"))
            malfunction_count += 1
            continue

        numeric_value = normalized.removesuffix("°C").replace(",", ".")
        try:
            parsed_values.append(float(numeric_value))
        except ValueError:
            parsed_values.append(float("nan"))
            invalid_rows.append(data.loc[index].to_dict())

    data["Temperature_Anomaly"] = parsed_values
    return data, pd.DataFrame(invalid_rows), malfunction_count


def sort_and_deduplicate(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sort observations by month and keep one row for each month."""
    sorted_data = data.sort_values("Date", kind="stable").reset_index(drop=True)
    duplicate_count = int(sorted_data.duplicated("Date").sum())
    deduplicated = sorted_data.drop_duplicates("Date", keep="first").reset_index(drop=True)
    return deduplicated, duplicate_count


def remove_iqr_outliers(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replace non-missing values outside the 1.5-IQR fences with NaN."""
    data = data.copy()
    valid_values = data["Temperature_Anomaly"].dropna()
    q1 = float(valid_values.quantile(0.25))
    q3 = float(valid_values.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_mask = data["Temperature_Anomaly"].notna() & (
        (data["Temperature_Anomaly"] < lower_fence)
        | (data["Temperature_Anomaly"] > upper_fence)
    )
    outlier_count = int(outlier_mask.sum())
    data.loc[outlier_mask, "Temperature_Anomaly"] = float("nan")
    statistics = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outlier_count": outlier_count,
    }
    return data, statistics


def write_monthly_checkpoint(data: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Write the current cleaned monthly state before normalization is available."""
    checkpoint = pd.DataFrame(
        {
            "date": data["Date"].dt.strftime("%Y-%m"),
            "anomaly_c": data["Temperature_Anomaly"],
            "z": float("nan"),
        }
    )
    checkpoint.to_csv(path, index=False, float_format="%.6f")


def main() -> None:
    data, footer = load_raw_data()
    data, unparsed, swapped_count = standardize_dates(data)
    data, invalid_values, malfunction_count = parse_anomalies(data)
    data, duplicate_count = sort_and_deduplicate(data)
    data, iqr_statistics = remove_iqr_outliers(data)
    write_monthly_checkpoint(data)
    print(f"Loaded candidate data rows: {len(data)}")
    print(f"Discarded file-hygiene rows: {len(footer)}")
    print(f"Swapped rows repaired: {swapped_count}")
    print(f"Unparsed date rows: {len(unparsed)}")
    print(f"Invalid anomaly rows: {len(invalid_values)}")
    print(f"Malfunction codes removed: {malfunction_count}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"IQR fences: {iqr_statistics['lower_fence']:.6f} to {iqr_statistics['upper_fence']:.6f}")
    print(f"IQR outliers removed: {iqr_statistics['outlier_count']}")
    print(f"Columns: {', '.join(data.columns)}")


if __name__ == "__main__":
    main()