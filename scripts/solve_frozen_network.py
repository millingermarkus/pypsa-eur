# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Dispatch the network of the current run with the capacities of another run.

Loads the composed (unsolved) network of the current run and the solved
network of a design run (`solving: frozen: design_network`). All capacities
that are extendable in the current run are fixed to the optimal capacities of
the design run; those that do not exist in the design run (e.g. assets of a
measure that the design run did not have) are set to zero. The network is
then only dispatched, so that changed capacities cannot compensate for
changed assumptions, e.g. a missing flexibility option or a different weather
year.

To keep the dispatch feasible, load shedding at the value of lost load and
spilling of surplus energy are added at all buses with loads (except CO2
buses). With `co2_price_from_design`, CO2 emissions may exceed the CO2 limit at
the CO2 price of the design run.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from pypsa.descriptors import nominal_attrs

from scripts._benchmark import memory_logger
from scripts._helpers import configure_logging, set_scenario_config
from scripts.solve_network import (
    collect_kwargs,
    create_optimization_model,
    prepare_network,
)

logger = logging.getLogger(__name__)

CO2_LIKE = ("co2", "emission")


def freeze_capacities(n: pypsa.Network, design: pypsa.Network) -> pd.DataFrame:
    """
    Fix the extendable capacities of `n` to the optimal capacities of `design`.

    Returns a summary per component and carrier of the capacities taken from
    the design, set to zero (only in `n`) and not used (only in `design`).
    """
    summary = []
    for c, attr in nominal_attrs.items():
        static = n.static(c)
        if static.empty or f"{attr}_extendable" not in static:
            continue
        ext = static.index[static[f"{attr}_extendable"]]
        d_static = design.static(c)
        d_ext = (
            d_static.index[d_static[f"{attr}_extendable"]]
            if f"{attr}_extendable" in d_static
            else pd.Index([])
        )
        common = ext.intersection(d_static.index)
        only_target = ext.difference(d_static.index)
        only_design = d_ext.difference(static.index)

        static.loc[common, attr] = d_static.loc[common, f"{attr}_opt"].clip(lower=0)
        static.loc[only_target, attr] = 0.0
        static[f"{attr}_extendable"] = False

        for label, idx, src in [
            ("from design", common, static.loc[common, attr]),
            ("only in run (set to 0)", only_target, None),
            ("only in design (not used)", only_design, d_static.loc[only_design, f"{attr}_opt"]),
        ]:
            if len(idx):
                carriers = (static if src is None or label == "from design" else d_static).loc[idx, "carrier"]
                cap = src if src is not None else pd.Series(0.0, index=idx)
                for carrier, v in cap.groupby(carriers.values).sum().items():
                    summary.append((c, label, carrier, int((carriers == carrier).sum()), v))
    summary = pd.DataFrame(summary, columns=["component", "status", "carrier", "count", "capacity"])
    for status, df in summary.groupby("status"):
        logger.info(
            f"Capacities {status}: {int(df['count'].sum())} components"
            + (f" ({', '.join(sorted(df.carrier.unique())[:15])})" if status != "from design" else "")
        )
    return summary


def add_shedding_and_spill(n: pypsa.Network, voll: float, spill_cost: float) -> pd.Index:
    """Add load shedding (at `voll`) and spilling (at `spill_cost`) at buses with loads."""
    buses = pd.Index(n.loads.bus.unique())
    carriers = n.buses.carrier.reindex(buses).fillna("")
    buses = buses[~carriers.str.contains("|".join(CO2_LIKE), case=False).values]
    n.add("Carrier", ["load shedding", "spill"])
    for suffix, carrier, sign, cost in [
        (" load shedding", "load shedding", 1.0, voll),
        (" spill", "spill", -1.0, spill_cost),
    ]:
        n.add(
            "Generator",
            buses + suffix,
            bus=buses.values,
            carrier=carrier,
            sign=sign,
            marginal_cost=cost,
            p_nom=1e7,
        )
    logger.info(
        f"Added load shedding ({voll} EUR/MWh) and spill ({spill_cost} EUR/MWh) at {len(buses)} buses"
    )
    return buses


def design_co2_price(design: pypsa.Network) -> float:
    """CO2 price (EUR/t) of the design run: shadow price of its upper CO2 limit."""
    glc = design.global_constraints
    upper = glc[(glc.type == "co2_atmosphere") & (glc.sense == "<=")]
    if upper.empty or "mu" not in glc:
        return np.nan
    return float(-upper.mu.iloc[0])


def add_co2_slack(n: pypsa.Network, price: float) -> None:
    """Allow CO2 emissions above the limit at `price` (EUR/t)."""
    if "co2 atmosphere" not in n.buses.index:
        logger.warning("No 'co2 atmosphere' bus; CO2 limit stays a hard limit.")
        return
    n.add("Carrier", "co2 slack")
    n.add("Bus", "co2 slack", carrier="co2 slack", unit="t_co2")
    n.add("Store", "co2 slack", bus="co2 slack", carrier="co2 slack", e_nom=1e12)
    n.add(
        "Link",
        "co2 atmosphere slack",
        bus0="co2 atmosphere",
        bus1="co2 slack",
        carrier="co2 slack",
        p_nom=1e9,
        marginal_cost=price,
    )
    logger.info(f"CO2 above the limit allowed at the design CO2 price of {price:.1f} EUR/t")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("solve_frozen_network", horizon=2050, run="less_flex")

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    cf_solving = snakemake.params.solving["options"]
    cf_frozen = snakemake.params.solving["frozen"]
    np.random.seed(cf_solving.get("seed", 123))
    planning_horizons = snakemake.wildcards["horizon"]

    n = pypsa.Network(snakemake.input.network)
    design = pypsa.Network(snakemake.input.design)
    logger.info(f"Freezing capacities from {snakemake.input.design}")

    summary = freeze_capacities(n, design)
    summary.to_csv(snakemake.output.capacity_summary, index=False)
    add_shedding_and_spill(n, cf_frozen["voll"], cf_frozen["spill_cost"])
    if cf_frozen["co2_price_from_design"]:
        price = design_co2_price(design)
        if np.isnan(price):
            logger.warning("No CO2 price in the design network; CO2 limit stays a hard limit.")
        else:
            add_co2_slack(n, max(price, 0.0))

    prepare_network(
        n,
        config=snakemake.config,
        solve_opts=cf_solving,
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params["sector"]["limit_max_growth"],
        rolling_horizon=False,
    )

    logging_frequency = snakemake.config.get("solving", {}).get("mem_logging_frequency", 30)
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        model_kwargs, solve_kwargs = collect_kwargs(
            snakemake.config,
            snakemake.params.solving,
            planning_horizons,
            log_fn=snakemake.log.solver,
            mode="single",
        )
        create_optimization_model(
            n,
            config=snakemake.config,
            params=snakemake.params,
            model_kwargs=model_kwargs,
            solve_kwargs=solve_kwargs,
            planning_horizons=planning_horizons,
            snakemake=snakemake,
        )
        status, condition = n.optimize.solve_model(**solve_kwargs)
    logger.info(f"Maximum memory usage: {mem.mem_usage}")
    if status != "ok":
        raise RuntimeError(f"Frozen dispatch not solved: {status}, {condition}")

    shed = n.generators.index[n.generators.carrier == "load shedding"]
    spill = n.generators.index[n.generators.carrier == "spill"]
    w = n.snapshot_weightings.generators
    shed_e = n.generators_t.p[shed].mul(w, axis=0).sum()
    spill_e = n.generators_t.p[spill].mul(w, axis=0).sum()
    slack = (
        n.links_t.p0["co2 atmosphere slack"].mul(w).sum()
        if "co2 atmosphere slack" in n.links.index
        else 0.0
    )
    carrier = n.buses.carrier
    pd.concat(
        {
            "load shedding MWh": shed_e.groupby(carrier.reindex(n.generators.loc[shed, "bus"]).values).sum(),
            "spill MWh": spill_e.groupby(carrier.reindex(n.generators.loc[spill, "bus"]).values).sum(),
        },
        axis=1,
    ).fillna(0.0).to_csv(snakemake.output.shedding)
    logger.info(
        f"Objective {n.objective / 1e9:.2f} bn EUR; load shedding {shed_e.sum() / 1e6:.3f} TWh; "
        f"spill {spill_e.sum() / 1e6:.3f} TWh; CO2 above limit {slack / 1e6:.2f} Mt"
    )

    n.meta["snakemake_config"] = dict(snakemake.config, wildcards=dict(snakemake.wildcards))
    n.export_to_netcdf(snakemake.output.network)
