# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the renewable liquid fuel share and solid biomass limit constraints."""

import pandas as pd
import pypsa
import pytest

from scripts.solve_network import (
    add_renewable_liquid_fuel_share_constraint,
    add_solid_biomass_limit_constraint,
)

FUEL_SHARE = {
    "share": 0.8,
    "sense": ">=",
    "producers": ["biomass to liquid", "Fischer-Tropsch"],
    "demand_carriers": ["kerosene for aviation", "shipping oil"],
}
SOLID_LIMIT = {
    "limit": 1e-6,  # TWh, i.e. 1 MWh per year
    "generator_carriers": ["solid biomass"],
    "link_carriers": ["solid biomass import"],
}


@pytest.fixture
def n():
    """Two snapshots with a weighting of 2 hours each."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-01", periods=2, freq="2h"))
    n.snapshot_weightings.loc[:, :] = 2.0

    for bus in ["oil", "H2", "solid biomass", "EU solid biomass import"]:
        n.add("Bus", bus)
    n.add("Generator", "fossil oil", bus="oil", p_nom=100, marginal_cost=1)
    n.add("Generator", "H2 supply", bus="H2", p_nom=100, marginal_cost=10)
    n.add(
        "Generator",
        "solid biomass",
        bus="solid biomass",
        carrier="solid biomass",
        p_nom=100,
        marginal_cost=1,
    )
    n.add(
        "Generator",
        "biomass import",
        bus="EU solid biomass import",
        p_nom=100,
        marginal_cost=0.5,
    )
    n.add(
        "Link",
        "BtL",
        bus0="solid biomass",
        bus1="oil",
        carrier="biomass to liquid",
        p_nom=100,
        efficiency=0.5,
    )
    n.add(
        "Link",
        "FT",
        bus0="H2",
        bus1="oil",
        carrier="Fischer-Tropsch",
        p_nom=100,
        efficiency=0.5,
    )
    n.add(
        "Link",
        "import",
        bus0="EU solid biomass import",
        bus1="solid biomass",
        carrier="solid biomass import",
        p_nom=100,
    )
    n.add("Load", "kerosene", bus="oil", carrier="kerosene for aviation", p_set=6)
    n.add("Load", "shipping", bus="oil", carrier="shipping oil", p_set=4)
    return n


def solve(n, constraint, config):
    n.optimize.create_model()
    constraint(n, n.snapshots, config)
    status, _ = n.optimize.solve_model(solver_name="highs")
    assert status == "ok"


def test_renewable_liquid_fuel_share(n):
    solve(n, add_renewable_liquid_fuel_share_constraint, FUEL_SHARE)

    w = n.snapshot_weightings.generators
    output = (n.links_t.p0[["BtL", "FT"]] * 0.5).mul(w, axis=0).sum().sum()
    demand = 10 * w.sum()
    assert output == pytest.approx(0.8 * demand)


def test_solid_biomass_limit_counts_imports(n):
    n.add("Load", "biomass demand", bus="solid biomass", p_set=0)
    n.links.loc["BtL", "p_min_pu"] = 0.1  # forces 10 MW of biomass input

    with pytest.raises(AssertionError):
        solve(n, add_solid_biomass_limit_constraint, SOLID_LIMIT)

    n.links.loc["BtL", "p_min_pu"] = 0.0
    solve(n, add_solid_biomass_limit_constraint, SOLID_LIMIT)
    w = n.snapshot_weightings.generators
    supply = (
        n.generators_t.p["solid biomass"].mul(w).sum()
        + n.links_t.p0["import"].mul(w).sum()
    )
    assert supply <= 1 + 1e-6
