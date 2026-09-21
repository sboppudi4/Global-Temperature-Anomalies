# Global Temperature Anomalies

I am building a reproducible Python pipeline for cleaning and analyzing a simulated monthly global temperature anomaly record. The dataset covers January 1880 through December 2025, and each value is measured in degrees Celsius relative to the 1901-2000 mean.

This is a teaching dataset, not an official NOAA, NASA GISS, or HadCRUT record. Its broad shape is inspired by published records, but the values should not be cited as official climate observations.

## What I am doing

The source CSV intentionally contains several data-quality problems. I am working through them in stages so that every transformation can be reproduced and explained:

- Load the raw fields as text and remove only clearly identified file-hygiene rows.
- Standardize mixed date formats, including the special two-digit-year rules.
- Repair rows where the date and temperature fields were swapped.
- Parse temperature values, missing-value tokens, and sensor-malfunction codes.
- Sort the observations and remove duplicate months.
- Detect outliers with the IQR rule and record the resulting fences.
- Restore the complete monthly timeline and fill missing months by time-based linear interpolation.
- Calculate the twentieth-century baseline difference and full-series z-score.
- Summarize the cleaned data by year and identify the five warmest years.
- Produce a dual-encoded chart whose position and color both represent the anomaly record.
- Generate a one-page IEEE-format LaTeX report containing the methods, results, table, and chart.

## Project files

The provided input file is `global_temp_dirty_v2.csv`. The Python pipeline is being developed incrementally in `clean.py`, with each meaningful stage validated and committed separately.

The finished project will also contain:

- `cleaned_monthly.csv`, with the date, cleaned anomaly, and z-score for every month.
- `cleaning_log.txt`, containing cleaning counts and any rows that could not be parsed.
- A vector chart or high-resolution PNG.
- `report.tex` and the compiled one-page `report.pdf`.
- A ZIP archive containing the complete submission.

## Current status

The first pipeline stage is complete. It loads the CSV as text, strips whitespace from headers and values, and separates four explicit footer or blank hygiene rows while preserving the remaining 1,707 candidate rows for later inspection. Date parsing and the remaining cleaning stages are next.

I am keeping the process deliberately incremental: each stage should be understandable on its own, easy to validate, and committed before the next stage changes the data.