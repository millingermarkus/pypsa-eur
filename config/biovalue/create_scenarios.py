# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create the biovalue scenarios from the run sets of pypsa-eur-sec (biovalue branch).

Writes ``config/biovalue/scenarios.yaml`` with one scenario per combination of
run set, CO2 limit, sequestration potential and solid biomass limit, and one
``config/biovalue/config.biovalue_<set>.yaml`` per run set that selects its
scenarios via ``run.name``.

The technology scenarios follow Tables 2 and 3 of the biovalue paper:

=================================  ========  ===========  =======  ===============
Technology scenario                run set   electricity  fuels    CCS
=================================  ========  ===========  =======  ===============
Optimistic                         base      optimistic   FT0/H20  CC3/CS0/high
Restricted VRE, costly nuclear     PE        restricted   FT0/H20  CC3/CS0/high
Costly fuel technologies           efuel     optimistic   FT2/H22  CC3/CS0/high
Costly CCS                         ccs       optimistic   FT0/H20  CC0/CS2/low
Pessimistic                        all       restricted   FT2/H22  CC0/CS2/low
=================================  ========  ===========  =======  ===============

Mapping to the configuration:

- grid extension (x2 / none): ``electricity.transmission_limit`` v2.0 / v1.0
- onshore wind and utility PV capacity per km2 (10 / 5 and 15 / 7 MW/km2):
  ``adjustments.sector.factor`` on ``p_nom_max``, relative to the
  ``capacity_per_sqkm`` in ``config/config.default.yaml``
- nuclear investment (4315 / 6932 EUR/kW): ``data/biovalue/costs_NUC0.csv`` /
  ``costs_NUC2.csv``
- electrolysis, Fischer-Tropsch and DAC (``FT0/FT2``, ``H20/H22``,
  ``CC3/CC0``): cost files in ``data/biovalue/``
- CO2 sequestration cost (``CS0/CS2``): ``sector.co2_sequestration_cost``
- capture efficiency (high 0.95 / low 0.8): ``sector.cc_fraction`` (SMR CC,
  waste CHP CC, bioH2) and the capture rates of all other carbon capture
  routes in ``data/biovalue/costs_CCeta_high.csv`` / ``costs_CCeta_low.csv``
- ``Co2Lx``: ``co2_budget.upper[2050]``
- ``seqN``: ``sector.co2_sequestration_potential[2050]``
- ``solidN``: ``sector.solid_biomass_limit.limit`` (TWh)

Usage (from the repository root)::

    python config/biovalue/create_scenarios.py
"""

from pathlib import Path

import yaml

HERE = Path(__file__).parent
DEFAULT_CONFIG = HERE.parent / "config.default.yaml"

# Table 3, electricity provision
ELECTRICITY = {
    "optimistic": dict(lv=2.0, onwind=10, solar=15, nuclear="NUC0"),
    "restricted": dict(lv=1.0, onwind=5, solar=7, nuclear="NUC2"),
}

# Table 2; run set names of the server configs config/biovalue/config.biovalue_<set>.yaml
RUN_SETS = {
    "base": dict(elec="optimistic", ft="FT0", h2="H20", cc="CC3", cs="CS0", eta="high"),
    "PE": dict(elec="restricted", ft="FT0", h2="H20", cc="CC3", cs="CS0", eta="high"),
    "efuel": dict(
        elec="optimistic", ft="FT2", h2="H22", cc="CC3", cs="CS0", eta="high"
    ),
    "ccs": dict(elec="optimistic", ft="FT0", h2="H20", cc="CC0", cs="CS2", eta="low"),
    "all": dict(elec="restricted", ft="FT2", h2="H22", cc="CC0", cs="CS2", eta="low"),
}

# Table 3, capture efficiency
CAPTURE_EFFICIENCY = {"high": 0.95, "low": 0.8}

# (Co2L relative to 1990, sequestration potential in Mt/a)
CO2_SEQ = [(-0.1, 650), (0.0, 650), (0.0, 200), (0.1, 650)]

# solid biomass limits in TWh/a
SOLID = [0, 50, 100, 150, 200, 250] + list(range(500, 3501, 250))

SEQUESTRATION_COST = {"CS0": 10, "CS2": 50}


def capacity_adjustments(onwind: float, solar: float) -> dict:
    """
    Factors on p_nom_max that turn the default capacity densities into the
    given ones (MW/km2). Utility PV covers both fixed-tilt and single-axis
    tracking, which keep their default ratio.
    """
    renewable = yaml.safe_load(DEFAULT_CONFIG.read_text())["renewable"]
    onwind_factor = onwind / renewable["onwind"]["capacity_per_sqkm"]
    solar_factor = solar / renewable["solar"]["capacity_per_sqkm"]
    return {
        "sector": {
            "factor": {
                "Generator": {
                    "onwind": {"p_nom_max": round(onwind_factor, 4)},
                    "solar": {"p_nom_max": round(solar_factor, 4)},
                    "solar-hsat": {"p_nom_max": round(solar_factor, 4)},
                }
            }
        }
    }


def co2_tag(value: float) -> str:
    return ("m" if value < 0 else "") + f"{abs(value):.1f}".replace(".", "p")


def scenario(settings: dict, co2: float, seq: int, solid: int) -> dict:
    elec = ELECTRICITY[settings["elec"]]
    cost_files = (
        [elec["nuclear"]]
        + [settings[k] for k in ("ft", "h2", "cc")]
        + [f"CCeta_{settings['eta']}"]
    )
    return {
        "electricity": {"transmission_limit": f"v{elec['lv']:.1f}"},
        "co2_budget": {"upper": {2050: co2}},
        "costs": {
            "custom_cost_fn": ["data/custom_costs.csv"]
            + [f"data/biovalue/costs_{f}.csv" for f in cost_files]
        },
        "sector": {
            "co2_sequestration_potential": {2050: seq},
            "co2_sequestration_cost": SEQUESTRATION_COST[settings["cs"]],
            "cc_fraction": CAPTURE_EFFICIENCY[settings["eta"]],
            "solid_biomass_limit": {"limit": solid},
        },
        "adjustments": capacity_adjustments(elec["onwind"], elec["solar"]),
    }


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def main() -> None:
    scenarios = {}
    run_names = {}
    for set_name, settings in RUN_SETS.items():
        names = []
        for co2, seq in CO2_SEQ:
            for solid in SOLID:
                name = f"{set_name}_co2{co2_tag(co2)}_seq{seq}_solid{solid}"
                scenarios[name] = scenario(settings, co2, seq, solid)
                names.append(name)
        run_names[set_name] = names

    header = (
        "# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>\n"
        "#\n"
        "# SPDX-License-Identifier: CC0-1.0\n"
        "\n"
        "# Generated by config/biovalue/create_scenarios.py, do not edit by hand.\n"
    )

    with open(HERE / "scenarios.yaml", "w") as f:
        f.write(header + "\n")
        yaml.dump(scenarios, f, Dumper=NoAliasDumper, sort_keys=False)

    for set_name, names in run_names.items():
        with open(HERE / f"config.biovalue_{set_name}.yaml", "w") as f:
            f.write(header + f"# Run set '{set_name}' of the biovalue runs.\n" + "\n")
            yaml.dump(
                {"run": {"name": names}}, f, Dumper=NoAliasDumper, sort_keys=False
            )

    print(f"Wrote {len(scenarios)} scenarios in {len(run_names)} run sets.")  # noqa: T201


if __name__ == "__main__":
    main()
