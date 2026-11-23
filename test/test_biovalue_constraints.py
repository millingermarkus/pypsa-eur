# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the renewable liquid fuel share and solid biomass limit constraints."""

import pandas as pd
import pypsa
import pytest

from scripts.prepare_sector_network import add_renewable_fuel_tracking
from scripts.solve_network import (
    add_renewable_liquid_fuel_share_constraint,
    add_solid_biomass_limit_constraint,
)

FUEL_SHARE = {
    "share": 0.8,
    "sense": ">=",
    "producers": ["biomass to liquid", "Fischer-Tropsch", "methanolisation"],
    "demand_carriers": ["kerosene for aviation", "shipping oil", "shipping methanol"],
}
SOLID_LIMIT = {
    "limit": 1e-6,  # TWh, i.e. 1 MWh per year
    "generator_carriers": ["solid biomass"],
    "link_carriers": ["solid biomass import"],
}


def add_demand(n, carrier, pool, p_set):
    """Fuel or feedstock demand supplied from a pooled bus, as in the sector network."""
    n.add("Bus", carrier, carrier=carrier)
    n.add("Load", carrier, bus=carrier, carrier=carrier, p_set=p_set)
    n.add(
        "Link",
        f"EU {carrier}",
        bus0=pool,
        bus1=carrier,
        bus2="co2 atmosphere",
        carrier=carrier,
        efficiency2=0.27,
        p_nom_extendable=True,
    )


@pytest.fixture
def n():
    """
    Two snapshots with a weighting of 2 hours each. Pooled oil and methanol
    buses supply fuels (10 MW oil, 5 MW methanol) and chemicals (20 MW naphtha,
    20 MW industry methanol).
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-01", periods=2, freq="2h"))
    n.snapshot_weightings.loc[:, :] = 2.0

    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add("Bus", "EU oil", carrier="oil", location="EU", unit="MWh_LHV")
    n.add("Bus", "EU methanol", carrier="methanol", location="EU", unit="MWh_LHV")
    for bus in ["H2", "solid biomass", "EU solid biomass import"]:
        n.add("Bus", bus)

    n.add("Generator", "fossil oil", bus="EU oil", p_nom=1000, marginal_cost=1)
    n.add(
        "Generator", "fossil methanol", bus="EU methanol", p_nom=1000, marginal_cost=1
    )
    n.add("Generator", "H2 supply", bus="H2", p_nom=1000, marginal_cost=10)
    n.add(
        "Generator",
        "solid biomass",
        bus="solid biomass",
        carrier="solid biomass",
        p_nom=1000,
        marginal_cost=1,
    )
    n.add(
        "Generator",
        "biomass import",
        bus="EU solid biomass import",
        p_nom=1000,
        marginal_cost=0.5,
    )
    n.add(
        "Link",
        "BtL",
        bus0="solid biomass",
        bus1="EU oil",
        carrier="biomass to liquid",
        p_nom=1000,
        efficiency=0.5,
    )
    n.add(
        "Link",
        "FT",
        bus0="H2",
        bus1="EU oil",
        carrier="Fischer-Tropsch",
        p_nom=1000,
        efficiency=0.5,
    )
    n.add(
        "Link",
        "methanolisation",
        bus0="H2",
        bus1="EU methanol",
        carrier="methanolisation",
        p_nom=1000,
        efficiency=0.8,
    )
    n.add(
        "Link",
        "import",
        bus0="EU solid biomass import",
        bus1="solid biomass",
        carrier="solid biomass import",
        p_nom=1000,
    )
    n.add("Store", "co2 atmosphere", bus="co2 atmosphere", e_nom_extendable=True)

    add_demand(n, "kerosene for aviation", "EU oil", 6)
    add_demand(n, "shipping oil", "EU oil", 4)
    add_demand(n, "naphtha for industry", "EU oil", 20)
    add_demand(n, "shipping methanol", "EU methanol", 5)
    add_demand(n, "industry methanol", "EU methanol", 20)
    return n


def solve(n, constraint, config):
    n.optimize.create_model()
    constraint(n, n.snapshots, config)
    status, _ = n.optimize.solve_model(solver_name="highs")
    assert status == "ok"


def test_tracking_reroutes_producers_and_copies_fuel_links(n):
    add_renewable_fuel_tracking(
        n, FUEL_SHARE["producers"], FUEL_SHARE["demand_carriers"]
    )

    assert n.links.at["BtL", "bus1"] == "EU oil renewable"
    assert n.links.at["methanolisation", "bus1"] == "EU methanol renewable"
    assert n.buses.at["EU oil renewable", "carrier"] == "oil"
    assert n.links.at["EU oil renewable blending", "bus1"] == "EU oil"

    copy = n.links.loc["EU kerosene for aviation renewable"]
    assert copy.bus0 == "EU oil renewable"
    assert copy.bus1 == "kerosene for aviation"
    assert copy.bus2 == "co2 atmosphere"
    assert copy.efficiency2 == 0.27

    # chemicals are only supplied from the pooled buses
    assert "EU naphtha for industry renewable" not in n.links.index
    assert "EU industry methanol renewable" not in n.links.index


def test_renewable_liquid_fuel_share_counts_fuel_deliveries(n):
    add_renewable_fuel_tracking(
        n, FUEL_SHARE["producers"], FUEL_SHARE["demand_carriers"]
    )
    solve(n, add_renewable_liquid_fuel_share_constraint, FUEL_SHARE)

    w = n.snapshot_weightings.generators
    deliveries = n.links.index[n.links.bus0.str.endswith(" renewable")]
    delivered = n.links_t.p0[deliveries].drop(
        columns=deliveries[deliveries.str.endswith("blending")]
    )
    assert delivered.mul(w, axis=0).sum().sum() == pytest.approx(0.8 * 15 * w.sum())


def test_renewable_fuel_for_chemicals_does_not_count(n):
    """
    Renewable production could cover 80% of fuel demand through the pooled
    buses, but only direct deliveries to fuel demand count.
    """
    add_renewable_fuel_tracking(
        n, FUEL_SHARE["producers"], FUEL_SHARE["demand_carriers"]
    )
    fuel_copies = n.links.index[
        n.links.bus0.str.endswith(" renewable")
        & ~n.links.index.str.endswith("blending")
    ]
    n.links.loc[fuel_copies, "p_nom_extendable"] = False
    n.links.loc[fuel_copies, "p_nom"] = 0

    with pytest.raises(AssertionError):
        solve(n, add_renewable_liquid_fuel_share_constraint, FUEL_SHARE)


@pytest.mark.parametrize("count_conversion", [True, False])
def test_methanol_to_kerosene_counts(n, count_conversion):
    """
    Renewable oil cannot reach fuel demand, so 70% of the 15 MW fuel demand
    (10.5 MW) needs renewable methanol for shipping (5 MW) plus renewable
    methanol converted to kerosene (up to 6 MW).
    """
    n.add(
        "Link",
        "EU methanol-to-kerosene",
        bus0="EU methanol",
        bus1="kerosene for aviation",
        carrier="methanol-to-kerosene",
        efficiency=0.9,
        p_nom_extendable=True,
    )
    conversion = ["methanol-to-kerosene"] if count_conversion else []
    config = FUEL_SHARE | {"share": 0.7, "conversion_carriers": conversion}
    add_renewable_fuel_tracking(
        n, config["producers"], config["demand_carriers"] + conversion
    )
    oil_copies = n.links.index[n.links.bus0 == "EU oil renewable"].drop(
        "EU oil renewable blending"
    )
    n.links.loc[oil_copies, "p_nom_extendable"] = False
    n.links.loc[oil_copies, "p_nom"] = 0

    if not count_conversion:
        with pytest.raises(AssertionError):
            solve(n, add_renewable_liquid_fuel_share_constraint, config)
        return

    solve(n, add_renewable_liquid_fuel_share_constraint, config)
    w = n.snapshot_weightings.generators
    kerosene = n.links_t.p0["EU methanol-to-kerosene renewable"].mul(0.9).mul(w).sum()
    methanol = n.links_t.p0["EU shipping methanol renewable"].mul(w).sum()
    assert kerosene + methanol == pytest.approx(0.7 * 15 * w.sum())


def test_solid_biomass_limit_counts_imports(n):
    n.add("Load", "biomass demand", bus="solid biomass", p_set=0)
    n.links.loc["BtL", "p_min_pu"] = 0.1  # forces 100 MW of biomass input

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
