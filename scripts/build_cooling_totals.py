# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build annual space cooling electricity demand per country from JRC-IDEES.

Reads the final energy consumption for space cooling of the residential
(`RES_hh_fec`) and tertiary (`SER_hh_fec`) sectors from JRC-IDEES for the
latest available year. Only electric space cooling is used (gas heat pumps
for cooling are negligible). Countries outside the EU27 are not covered by
JRC-IDEES and are left out; [prepare_sector_network][] estimates them from
the EU average cooling per person and cooling degree hour.

Outputs
-------
- `resources/cooling_totals.csv`: space cooling electricity in TWh/a per
  country (index) for `residential` and `services` (columns), and the data
  year.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

KTOE_TO_TWH = 11.63 / 1e3
IDEES_RENAME = {"GR": "EL", "GB": "UK"}
EU27 = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR",
    "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI",
    "SK",
]  # fmt: skip


def _cooling_row(fn: Path, sheet: str) -> pd.Series:
    """Return the electric space cooling row (ktoe per year) of a JRC-IDEES sheet."""
    df = pd.read_excel(fn, sheet, index_col=0)
    df = df.loc[:, [c for c in df.columns if isinstance(c, int | float) and c > 1990]]
    df.columns = df.columns.astype(int)
    index = df.index.astype(str).str.strip()
    for label in ("Electric space cooling", "Space cooling"):
        rows = df[index == label]
        if not rows.empty:
            return rows.iloc[0].astype(float)
    raise KeyError(f"No space cooling row in {fn.name} / {sheet}")


def cooling_per_country(ct: str, base_dir: str) -> pd.DataFrame:
    """Space cooling electricity (TWh/a) of the residential and tertiary sector per year."""
    ct_idees = IDEES_RENAME.get(ct, ct)
    root = Path(base_dir, ct_idees)
    files = {}
    for sector in ("Residential", "Tertiary"):
        candidates = sorted(root.glob(f"JRC-IDEES-*_{sector}_{ct_idees}.xlsx"))
        if not candidates:
            raise FileNotFoundError(f"No JRC-IDEES {sector} file for {ct} in {root}")
        files[sector] = candidates[-1]
    return pd.DataFrame(
        {
            "residential": _cooling_row(files["Residential"], "RES_hh_fec"),
            "services": _cooling_row(files["Tertiary"], "SER_hh_fec"),
        }
    ).mul(KTOE_TO_TWH)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_cooling_totals")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    countries = [c for c in snakemake.params.countries if c in EU27]
    totals = {}
    for ct in countries:
        df = cooling_per_country(ct, snakemake.input.idees)
        year = df.dropna().index.max()
        totals[ct] = df.loc[year]
        totals[ct]["year"] = year
    totals = pd.DataFrame(totals).T.rename_axis("country")
    totals["year"] = totals["year"].astype(int)

    missing = sorted(set(snakemake.params.countries) - set(totals.index))
    logger.info(
        f"Space cooling electricity {totals[['residential', 'services']].sum().sum():.1f} TWh/a "
        f"in {len(totals)} countries (JRC-IDEES {totals['year'].max()}); "
        f"estimated from the EU average later: {missing}"
    )
    totals.to_csv(snakemake.output.cooling_totals)
