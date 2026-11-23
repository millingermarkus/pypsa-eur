# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for DAC heat from industrial steam and electricity-only biomass power."""

from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

from scripts.prepare_sector_network import add_biomass_power, add_dac

NODES = pd.Index(["DE0 0", "FR0 0"])


@pytest.fixture
def costs():
    c = pd.DataFrame(
        {
            "capital_cost": [600.0, 1000.0, 1500.0, 2000.0],
            "efficiency": [1.0, 0.3, 0.25, 1.0],
            "VOM": [0.0, 3.0, 5.0, 0.0],
            "lifetime": [20.0, 25.0, 25.0, 25.0],
        },
        index=[
            "direct air capture",
            "biomass EOP",
            "central solid biomass CHP CC",
            "biomass CHP capture",
        ],
    )
    c.loc[
        "direct air capture", ["electricity-input", "compression-electricity-input"]
    ] = [0.4, 0.15]
    c.loc["direct air capture", ["heat-input", "compression-heat-output"]] = [1.6, 0.2]
    c.loc[
        "biomass CHP capture", ["electricity-input", "compression-electricity-input"]
    ] = [0.02, 0.08]
    c.loc["biomass CHP capture", "capture_rate"] = 0.8
    c.loc["solid biomass", "CO2 intensity"] = 0.4
    return c


@pytest.fixture
def spatial():
    return SimpleNamespace(
        biomass=SimpleNamespace(
            df=pd.DataFrame({"nodes": NODES + " solid biomass"}, index=NODES)
        ),
        co2=SimpleNamespace(df=pd.DataFrame({"nodes": "co2 stored"}, index=NODES)),
    )


@pytest.fixture
def n():
    n = pypsa.Network()
    n.add("Bus", NODES, location=NODES)
    n.add("Bus", NODES + " solid biomass", location=NODES)
    n.add(
        "Bus",
        NODES + " heat100-200 industry",
        location=NODES,
        carrier="heat100-200 industry",
    )
    # district heating only in DE
    n.add(
        "Bus",
        "DE0 0 urban central heat",
        location="DE0 0",
        carrier="urban central heat",
    )
    n.add("Bus", ["co2 atmosphere", "co2 stored"])
    return n


def test_dac_from_industry_steam(n, costs, spatial):
    options = {"dac_heat_source": "industry steam", "use_dac_waste_heat": 0.5}
    add_dac(n, costs, spatial, options)

    dac = n.links.query("carrier == 'DAC'")
    assert list(dac.index) == ["DE0 0 DAC", "FR0 0 DAC"]
    assert list(dac.bus1) == list(NODES + " heat100-200 industry")
    # full heat input of 1.6 MWh/tCO2 per 0.55 MWh electricity
    assert dac.efficiency.tolist() == pytest.approx([-1.6 / 0.55] * 2)
    # half of the compression heat to district heating where it exists
    assert dac.at["DE0 0 DAC", "bus4"] == "DE0 0 urban central heat"
    assert dac.at["DE0 0 DAC", "efficiency4"] == pytest.approx(0.1 / 0.55)
    assert dac.at["FR0 0 DAC", "bus4"] == ""
    assert dac.at["FR0 0 DAC", "efficiency4"] == 0


def test_dac_from_industry_steam_requires_steam_bus(costs, spatial):
    n = pypsa.Network()
    n.add("Bus", NODES, location=NODES)
    with pytest.raises(ValueError, match="industrial steam"):
        add_dac(n, costs, spatial, {"dac_heat_source": "industry steam"})


def test_dac_default_uses_low_temperature_heat(n, costs, spatial):
    add_dac(n, costs, spatial)
    dac = n.links.query("carrier == 'DAC'")
    assert list(dac.bus1) == ["DE0 0 urban central heat"]
    assert dac.efficiency.iloc[0] == pytest.approx(-1.4 / 0.55)


@pytest.mark.parametrize(
    "technology_cc", ["biomass EOP", "central solid biomass CHP CC"]
)
def test_biomass_power(n, costs, spatial, technology_cc):
    config = {
        "enable": True,
        "cc": True,
        "technology": "biomass EOP",
        "technology_cc": technology_cc,
        "capital_cost_factor": 0.5,
        "efficiency_factor": 1.2,
    }
    add_biomass_power(n, costs, spatial, NODES, config)

    plant = n.links.loc["DE0 0 solid biomass to electricity"]
    assert plant.bus0 == "DE0 0 solid biomass"
    assert plant.efficiency == pytest.approx(0.36)
    # capital cost per MW biomass input: 0.5 x 1000 per MW_el x 0.36
    assert plant.capital_cost == pytest.approx(180)

    cc = n.links.loc["FR0 0 solid biomass to electricity CC"]
    eff = 1.2 * costs.at[technology_cc, "efficiency"]
    if technology_cc == "biomass EOP":
        eff -= 0.4 * (0.02 + 0.08)  # capture electricity deducted
    assert cc.efficiency == pytest.approx(eff)
    assert cc.efficiency2 == pytest.approx(-0.32)  # 0.4 tCO2/MWh x 0.8
    assert cc.efficiency3 == pytest.approx(0.32)
    assert cc.capital_cost == pytest.approx(
        0.5
        * costs.at[technology_cc, "capital_cost"]
        * 1.2
        * costs.at[technology_cc, "efficiency"]
        + 2000 * 0.4
    )
