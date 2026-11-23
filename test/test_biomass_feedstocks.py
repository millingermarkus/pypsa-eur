# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for biomass feedstocks and the stepped biomass import supply curve."""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.build_biomass_potentials import build_feedstocks, uniform_feedstock_costs
from scripts.prepare_sector_network import (
    add_biomass_feedstock_generators,
    biomass_import_supply_curve,
)

CLASSES = {
    "solid biomass": ["Fuelwood residues", "Sawdust"],
    "biogas": ["Manure solid, liquid"],
    "not included": ["Poplar"],
}
GROUPS = {
    "forest residues": ["Fuelwood residues"],
    "industry wood residues": ["Sawdust"],
    "manure": ["Manure solid, liquid"],
}


@pytest.fixture
def commodities():
    """Potentials in TWh by region and ENSPRESO commodity."""
    return pd.DataFrame(
        {
            "Fuelwood residues": [3.0, 1.0],
            "Sawdust": [1.0, 0.0],
            "Manure solid, liquid": [2.0, 2.0],
            "Poplar": [5.0, 5.0],
        },
        index=["DE0 0", "FR0 0"],
    )


@pytest.fixture
def classes():
    """Final class potentials in MWh/a, halved by the sustainable share."""
    return pd.DataFrame(
        {"solid biomass": [2e6, 0.5e6], "biogas": [1e6, 1e6]},
        index=["DE0 0", "FR0 0"],
    )


@pytest.fixture
def costs():
    """Costs in EUR/MWh by country; FR lacks a cost for fuelwood residues."""
    return pd.DataFrame(
        {
            "Fuelwood residues": [10.0, np.nan],
            "Sawdust": [20.0, 30.0],
            "Manure solid, liquid": [5.0, 7.0],
        },
        index=["DE", "FR"],
    )


def test_build_feedstocks_splits_class_potentials(commodities, classes, costs):
    df = build_feedstocks(commodities, classes, costs, CLASSES, GROUPS)
    df = df.set_index(["region", "feedstock"])

    # feedstock potentials add up to the class potentials
    assert df.loc["DE0 0", "potential"].sum() == pytest.approx(3e6)
    assert df.at[("DE0 0", "forest residues"), "potential"] == pytest.approx(1.5e6)
    assert df.at[("DE0 0", "industry wood residues"), "potential"] == pytest.approx(
        0.5e6
    )
    assert df.at[("FR0 0", "industry wood residues"), "potential"] == 0
    assert df.at[("FR0 0", "manure"), "class"] == "biogas"

    # missing country costs are filled with the European average
    assert df.at[("FR0 0", "forest residues"), "cost"] == pytest.approx(10.0)
    assert df.at[("FR0 0", "industry wood residues"), "cost"] == pytest.approx(30.0)


def test_uniform_feedstock_costs(commodities, classes, costs):
    df = uniform_feedstock_costs(
        build_feedstocks(commodities, classes, costs, CLASSES, GROUPS)
    ).set_index(["region", "feedstock"])

    # forest residues: 1.5e6 MWh at 10 EUR/MWh (DE) and 0.5e6 MWh at 10 (FR, filled)
    assert df.xs("forest residues", level="feedstock").cost.tolist() == [10.0, 10.0]
    # manure: 1e6 MWh at 5 EUR/MWh (DE) and 1e6 MWh at 7 EUR/MWh (FR)
    assert df.xs("manure", level="feedstock").cost.tolist() == pytest.approx([6, 6])


def test_build_feedstocks_requires_full_coverage(commodities, classes, costs):
    groups = {k: v for k, v in GROUPS.items() if k != "industry wood residues"}
    with pytest.raises(ValueError, match="Missing: \\['Sawdust'\\]"):
        build_feedstocks(commodities, classes, costs, CLASSES, groups)


def test_build_feedstocks_rejects_mixed_classes(commodities, classes, costs):
    groups = {
        "mixed": ["Fuelwood residues", "Manure solid, liquid"],
        "industry wood residues": ["Sawdust"],
    }
    with pytest.raises(ValueError, match="exactly one"):
        build_feedstocks(commodities, classes, costs, CLASSES, groups)


@pytest.fixture
def feedstocks():
    return pd.DataFrame(
        {
            "region": ["DE0 0", "FR0 0", "DE0 0", "FR0 0"],
            "feedstock": ["forest residues"] * 2 + ["industry wood residues"] * 2,
            "class": ["solid biomass"] * 4,
            "potential": [3.0, 1.0, 2.0, 0.0],
            "cost": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_feedstock_generators_aggregate_to_eu_bus(feedstocks):
    n = pypsa.Network()
    n.add("Bus", "EU solid biomass")

    add_biomass_feedstock_generators(
        n, feedstocks, "solid biomass", ["EU solid biomass"]
    )

    gens = n.generators
    assert set(gens.index) == {
        "EU solid biomass forest residues",
        "EU solid biomass industry wood residues",
    }
    assert (gens.carrier == "solid biomass").all()
    assert gens.at["EU solid biomass forest residues", "e_sum_max"] == 4.0
    # potential-weighted cost
    assert gens.at[
        "EU solid biomass forest residues", "marginal_cost"
    ] == pytest.approx(12.5)


def test_feedstock_generators_per_region_skip_zero_potential(feedstocks):
    n = pypsa.Network()
    buses = pd.Index(["DE0 0 solid biomass", "FR0 0 solid biomass"])
    n.add("Bus", buses)

    add_biomass_feedstock_generators(n, feedstocks, "solid biomass", buses)

    assert "FR0 0 solid biomass industry wood residues" not in n.generators.index
    assert n.generators.at["FR0 0 solid biomass forest residues", "bus"] == (
        "FR0 0 solid biomass"
    )


def test_supply_curve_fills_up_to_threshold():
    tiers = biomass_import_supply_curve(
        domestic_potential=15,
        price=54,
        total_supply_threshold=20,
        step_size=1,
        n_steps=3,
        price_increment=0.9,
    )
    assert tiers.amount.tolist() == [5, 1, 1, 1]
    assert tiers.price.tolist() == pytest.approx([54, 54.9, 55.8, 56.7])


def test_supply_curve_deducts_domestic_excess():
    tiers = biomass_import_supply_curve(
        domestic_potential=21.5,
        price=54,
        total_supply_threshold=20,
        step_size=1,
        n_steps=3,
        price_increment=0.9,
    )
    # 1.5 above the threshold empties tier 2 and halves tier 3
    assert tiers.index.tolist() == [3, 4]
    assert tiers.amount.tolist() == pytest.approx([0.5, 1])
    assert tiers.price.tolist() == pytest.approx([55.8, 56.7])
