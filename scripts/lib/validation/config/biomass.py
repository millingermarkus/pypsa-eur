# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Biomass configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#biomass_cf
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scripts.lib.validation.config._base import ConfigModel


class _BiomassClassesConfig(ConfigModel):
    """Configuration for `biomass.classes` settings."""

    solid_biomass: list[str] = Field(
        default_factory=lambda: [
            "Agricultural waste",
            "Fuelwood residues",
            "Secondary Forestry residues - woodchips",
            "Sawdust",
            "Residues from landscape care",
        ],
        alias="solid biomass",
        description="The comodity that are included as solid biomass.",
    )
    not_included: list[str] = Field(
        default_factory=lambda: [
            "Sugar from sugar beet",
            "Rape seed",
            "Sunflower, soya seed ",
            "Bioethanol barley, wheat, grain maize, oats, other cereals and rye",
            "Miscanthus, switchgrass, RCG",
            "Willow",
            "Poplar",
            "FuelwoodRW",
            "C&P_RW",
        ],
        alias="not included",
        description="The comodity that are not included as a biomass potential.",
    )
    biogas: list[str] = Field(
        default_factory=lambda: [
            "Manure solid, liquid",
            "Sludge",
        ],
        description="The comodity that are included as biogas.",
    )
    municipal_solid_waste: list[str] = Field(
        default_factory=lambda: [
            "Municipal waste",
        ],
        alias="municipal solid waste",
        description="The commodities that are included as municipal solid waste.",
    )

    model_config = ConfigDict(populate_by_name=True)


class _BiomassFeedstocksConfig(ConfigModel):
    """Configuration for `biomass.feedstocks` settings."""

    enable: bool = Field(
        False,
        description="Split the `solid biomass` and `biogas` supply into one generator per feedstock, each with its own potential and its own cost from the JRC ENSPRESO cost sheet ('COST - NUTS0 EnergyCom'). Generators keep the carrier of their class and are named after the feedstock.",
    )
    cost_year: int | None = Field(
        None,
        ge=2010,
        le=2050,
        description="Year of the ENSPRESO feedstock costs. Defaults to `biomass.year` (overnight) or the planning horizon (myopic/perfect).",
    )
    cost_scenario: Literal["ENS_BaU_GFTM", "ENS_Low", "ENS_Med", "ENS_High"] | None = (
        Field(
            None,
            description="ENSPRESO scenario of the feedstock costs. Defaults to `biomass.scenario`.",
        )
    )
    groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Feedstock name mapped to the ENSPRESO commodities it consists of. Together, the feedstocks must cover exactly the commodities of the `solid biomass` and `biogas` classes in `biomass.classes`, and each feedstock must lie within one class.",
    )


class BiomassConfig(BaseModel):
    """Configuration for `biomass` settings."""

    year: int = Field(
        2030,
        ge=2010,
        le=2050,
        description="Year for which to retrieve biomass potential according to the assumptions of the `JRC ENSPRESO <https://data.jrc.ec.europa.eu/dataset/74ed5a04-7d74-4807-9eab-b94774309d9f>`_.",
    )
    scenario: Literal["ENS_Low", "ENS_Med", "ENS_High"] = Field(
        "ENS_Med",
        description="Scenario for which to retrieve biomass potential. The scenario definition can be seen in `ENSPRESO_BIOMASS <https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/ENSPRESO/ENSPRESO_BIOMASS.xlsx>`_.",
    )
    classes: _BiomassClassesConfig = Field(
        default_factory=_BiomassClassesConfig,
        description="Classification of biomass commodities.",
    )
    class_scenarios: dict[str, Literal["ENS_Low", "ENS_Med", "ENS_High"]] = Field(
        default_factory=dict,
        description="Use a different ENSPRESO potential scenario for the commodities of the given classes, e.g. `{biogas: ENS_Med}` while `biomass.scenario` is `ENS_High`.",
    )
    feedstocks: _BiomassFeedstocksConfig = Field(
        default_factory=_BiomassFeedstocksConfig,
        description="Feedstock-specific biomass supply with separate costs.",
    )
    share_unsustainable_use_retained: dict[int, float] = Field(
        default_factory=lambda: {
            2020: 1,
            2025: 1,
            2030: 0.66,
            2035: 0.33,
            2040: 0,
            2045: 0,
            2050: 0,
        },
        description="Share of unsustainable biomass use retained using primary production of Eurostat data as reference.",
    )
    share_sustainable_potential_available: dict[int, float] = Field(
        default_factory=lambda: {
            2020: 0,
            2025: 0,
            2030: 0.33,
            2035: 0.66,
            2040: 1,
            2045: 1,
            2050: 1,
        },
        description="Share determines phase-in of ENSPRESO biomass potentials.",
    )
