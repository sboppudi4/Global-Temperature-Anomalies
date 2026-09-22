"""Initial loading and file-hygiene stage for the temperature pipeline."""

from pathlib import Path
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm


INPUT_PATH = Path(__file__).with_name("global_temp_dirty_v2.csv")
OUTPUT_PATH = Path(__file__).with_name("cleaned_monthly.csv")
ANNUAL_OUTPUT_PATH = Path(__file__).with_name("annual_summary.csv")
FIGURE_PATH = Path(__file__).with_name("temperature_anomalies.pdf")
START_DATE = "1880-01-01"
END_DATE = "2025-12-01"
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


def complete_monthly_series(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reindex to the required monthly grid and interpolate missing anomalies."""
    full_index = pd.date_range(START_DATE, END_DATE, freq="MS")
    indexed = data.set_index("Date").reindex(full_index)
    missing_before = int(indexed["Temperature_Anomaly"].isna().sum())
    absent_months = len(full_index) - len(data)
    missing_observed_values = missing_before - absent_months
    indexed["Temperature_Anomaly"] = indexed["Temperature_Anomaly"].interpolate(method="time")
    missing_after = int(indexed["Temperature_Anomaly"].isna().sum())
    completed = indexed.reset_index(names="Date")
    statistics = {
        "expected_months": len(full_index),
        "observed_months": len(data),
        "absent_months": absent_months,
        "missing_observed_values": missing_observed_values,
        "months_imputed": missing_before - missing_after,
        "missing_after_interpolation": missing_after,
    }
    return completed, statistics


def normalize_series(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calculate the twentieth-century baseline difference and full-series z-score."""
    data = data.copy()
    baseline_mask = data["Date"].dt.year.between(1901, 2000)
    mu_20 = float(data.loc[baseline_mask, "Temperature_Anomaly"].mean())
    mu = float(data["Temperature_Anomaly"].mean())
    sigma = float(data["Temperature_Anomaly"].std(ddof=0))
    data["d"] = data["Temperature_Anomaly"] - mu_20
    data["z"] = (data["Temperature_Anomaly"] - mu) / sigma
    return data, {"mu_20": mu_20, "mu": mu, "sigma": sigma}


def summarize_annual(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate annual anomaly and z-score means and return the five warmest years."""
    annual = (
        data.assign(year=data["Date"].dt.year)
        .groupby("year", as_index=False)
        .agg(
            mean_anomaly_c=("Temperature_Anomaly", "mean"),
            mean_z=("z", "mean"),
            months=("Date", "count"),
        )
    )
    warmest = annual.nlargest(5, "mean_anomaly_c").reset_index(drop=True)
    return annual, warmest


def write_monthly_checkpoint(data: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Write the current cleaned monthly state before normalization is available."""
    checkpoint = pd.DataFrame(
        {
            "date": data["Date"].dt.strftime("%Y-%m"),
            "anomaly_c": data["Temperature_Anomaly"],
            "z": data["z"] if "z" in data else float("nan"),
        }
    )
    checkpoint.to_csv(path, index=False, float_format="%.6f")


def write_annual_summary(annual: pd.DataFrame, path: Path = ANNUAL_OUTPUT_PATH) -> None:
    """Write annual results for inspection and report generation."""
    annual.to_csv(path, index=False, float_format="%.6f")


def create_dual_encoded_chart(
    data: pd.DataFrame,
    mu_20: float,
    path: Path = FIGURE_PATH,
) -> None:
    """Create a segment-colored anomaly line with color centered at the baseline."""
    dates = data["Date"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    anomalies = data["Temperature_Anomaly"].to_numpy(dtype=float)
    baseline_differences = data["d"].to_numpy(dtype=float)
    points = np.column_stack([dates, anomalies]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    limit = max(abs(float(baseline_differences.min())), abs(float(baseline_differences.max())))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    figure, axis = plt.subplots(figsize=(3.5, 2.35))
    collection = LineCollection(segments, cmap="RdBu_r", norm=norm, linewidth=0.65)
    collection.set_array((baseline_differences[:-1] + baseline_differences[1:]) / 2)
    axis.add_collection(collection)
    axis.axhline(0, color="black", linewidth=0.45, linestyle="--")
    axis.set_xlim(dates.min(), dates.max())
    axis.set_ylim(anomalies.min() - 0.08, anomalies.max() + 0.08)
    axis.set_xlabel("Year")
    axis.set_ylabel("Temperature anomaly (°C)")
    axis.set_title("Simulated global temperature anomalies, 1880-2025")
    axis.grid(True, linewidth=0.25, alpha=0.35)
    colorbar = figure.colorbar(collection, ax=axis, pad=0.02)
    colorbar.set_label("d = anomaly - 1901-2000 mean (°C)")
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    data, footer = load_raw_data()
    data, unparsed, swapped_count = standardize_dates(data)
    data, invalid_values, malfunction_count = parse_anomalies(data)
    data, duplicate_count = sort_and_deduplicate(data)
    data, iqr_statistics = remove_iqr_outliers(data)
    data, interpolation_statistics = complete_monthly_series(data)
    data, normalization_statistics = normalize_series(data)
    write_monthly_checkpoint(data)
    annual, warmest = summarize_annual(data)
    write_annual_summary(annual)
    create_dual_encoded_chart(data, normalization_statistics["mu_20"])
    print(f"Loaded candidate data rows: {len(data)}")
    print(f"Discarded file-hygiene rows: {len(footer)}")
    print(f"Swapped rows repaired: {swapped_count}")
    print(f"Unparsed date rows: {len(unparsed)}")
    print(f"Invalid anomaly rows: {len(invalid_values)}")
    print(f"Malfunction codes removed: {malfunction_count}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"IQR fences: {iqr_statistics['lower_fence']:.6f} to {iqr_statistics['upper_fence']:.6f}")
    print(f"IQR outliers removed: {iqr_statistics['outlier_count']}")
    print(f"Absent months: {interpolation_statistics['absent_months']}")
    print(f"Missing observed values: {interpolation_statistics['missing_observed_values']}")
    print(f"Months imputed: {interpolation_statistics['months_imputed']}")
    print(f"Missing after interpolation: {interpolation_statistics['missing_after_interpolation']}")
    print(f"mu_20: {normalization_statistics['mu_20']:.6f}")
    print(f"mu: {normalization_statistics['mu']:.6f}")
    print(f"sigma: {normalization_statistics['sigma']:.6f}")
    print("Five warmest years:")
    print(warmest[["year", "mean_anomaly_c", "mean_z"]].to_string(index=False))
    print(f"Columns: {', '.join(data.columns)}")


if __name__ == "__main__":
    main()