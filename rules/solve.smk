# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


rule solve_network:
    """Optimises investment and dispatch for one planning horizon to minimise total system cost."""
    input:
        network=resources("networks/composed_{horizon}.nc"),
    output:
        network=RESULTS + "networks/solved_{horizon}.nc",
        model=(
            RESULTS + "models/solved_{horizon}.nc"
            if config["solving"]["options"]["store_model"]
            else []
        ),
    log:
        solver=normpath(RESULTS + "logs/solve_network/solver_{horizon}.log"),
        memory=RESULTS + "logs/solve_network/memory_{horizon}.log",
        python=RESULTS + "logs/solve_network/python_{horizon}.log",
    benchmark:
        (RESULTS + "benchmarks/solve_network_{horizon}.log")
    shadow:
        shadow_config
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        planning_horizons=config_provider("planning_horizons"),
        sector=config_provider("sector"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential"
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    script:
        scripts("solve_network.py")


rule solve_operations_network:
    """Re-optimises dispatch only, holding the capacities from the solved network fixed."""
    input:
        network=RESULTS + "networks/solved_{horizon}.nc",
    output:
        network=RESULTS + "networks/operations_{horizon}.nc",
    log:
        solver=normpath(RESULTS + "logs/solve_operations_network/solver_{horizon}.log"),
        memory=RESULTS + "logs/solve_operations_network/memory_{horizon}.log",
        python=RESULTS + "logs/solve_operations_network/python_{horizon}.log",
    benchmark:
        (RESULTS + "benchmarks/solve_operations_network_{horizon}.log")
    shadow:
        shadow_config
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        planning_horizons=config_provider("planning_horizons"),
        sector=config_provider("sector"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential"
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    script:
        scripts("solve_operations_network.py")


def input_frozen_design(w):
    """Solved network of the design run for rule `solve_frozen_network`."""
    path = config_provider("solving", "frozen", "design_network")(w)
    if not path:
        raise ValueError(
            "Rule solve_frozen_network needs `solving: frozen: design_network`."
        )
    return ancient(path.format(horizon=w.horizon))


rule solve_frozen_network:
    """Dispatches the network of this run with the capacities of a design run fixed."""
    input:
        network=resources("networks/composed_{horizon}.nc"),
        design=input_frozen_design,
    output:
        network=RESULTS + "networks/frozen_{horizon}.nc",
        capacity_summary=RESULTS + "csvs/frozen_capacities_{horizon}.csv",
        shedding=RESULTS + "csvs/frozen_shedding_{horizon}.csv",
    log:
        solver=normpath(RESULTS + "logs/solve_frozen_network/solver_{horizon}.log"),
        memory=RESULTS + "logs/solve_frozen_network/memory_{horizon}.log",
        python=RESULTS + "logs/solve_frozen_network/python_{horizon}.log",
    benchmark:
        (RESULTS + "benchmarks/solve_frozen_network_{horizon}.log")
    shadow:
        shadow_config
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        planning_horizons=config_provider("planning_horizons"),
        sector=config_provider("sector"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential"
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    script:
        scripts("solve_frozen_network.py")
