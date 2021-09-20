# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# coding: utf-8
"""
Retrieves conventional powerplant capacities and locations from `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_, assigns these to buses and creates a ``.csv`` file. It is possible to amend the powerplant database with custom entries provided in ``data/custom_powerplants.csv``.

Relevant Settings
-----------------

.. code:: yaml

    electricity:
      powerplants_filter:
      custom_powerplants:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity`

Inputs
------

- ``networks/base.nc``: confer :ref:`base`.
- ``data/custom_powerplants.csv``: custom powerplants in the same format as `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_ provides

Outputs
-------

- ``resource/powerplants.csv``: A list of conventional power plants (i.e. neither wind nor solar) with fields for name, fuel type, technology, country, capacity in MW, duration, commissioning year, retrofit year, latitude, longitude, and dam information as documented in the `powerplantmatching README <https://github.com/FRESNA/powerplantmatching/blob/master/README.md>`_; additionally it includes information on the closest substation/bus in ``networks/base.nc``.

    .. image:: ../img/powerplantmatching.png
        :scale: 30 %

    **Source:** `powerplantmatching on GitHub <https://github.com/FRESNA/powerplantmatching>`_

Description
-----------

The configuration options ``electricity: powerplants_filter`` and ``electricity: custom_powerplants`` can be used to control whether data should be retrieved from the original powerplants database or from custom amendmends. These specify `pandas.query <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_ commands.

1. Adding all powerplants from custom:

    .. code:: yaml

        powerplants_filter: false
        custom_powerplants: true

2. Replacing powerplants in e.g. Germany by custom data:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: true

    or

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: Country in ['Germany']


3. Adding additional built year constraints:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany'] and YearCommissioned <= 2015
        custom_powerplants: YearCommissioned <= 2015

"""

import logging
from _helpers import configure_logging

import pypsa
import powerplantmatching as pm
import pandas as pd
import numpy as np

from scipy.spatial import cKDTree as KDTree

from six import iteritems

logger = logging.getLogger(__name__)


def add_custom_powerplants(ppl):
    custom_ppl_query = snakemake.config['electricity']['custom_powerplants']
    if not custom_ppl_query:
        return ppl
    add_ppls = pd.read_csv(snakemake.input.custom_powerplants, index_col=0,
                           dtype={'bus': 'str'})
    if isinstance(custom_ppl_query, str):
        add_ppls.query(custom_ppl_query, inplace=True)
    return ppl.append(add_ppls, sort=False, ignore_index=True, verify_integrity=True)


def correct_nuclear_data(ppl):
    # Data from https://en.wikipedia.org/wiki/List_of_commercial_nuclear_reactors
    YearCommercial = {
        'St laurent': 1983,
        'Gravelines': 1985,
        'Paluel': 1986,
        'Penly': 1992,
        'Nogent': 1989,
        'Golfech': 1994,
        'St alban': 1987,
        'Belleville': 1989,
        'Blayais': 1983,
        'Cruas': 1985,
        'Fessenheim': 1978,
        'Flamanville': 2005,  # new reactor being built, others 1987
        'Dampierre': 1981,
        'Chinon': 1988,
        'Bugey': 1980,
        'Cattenom': 1992,
        'Chooz': 2000,
        'Civaux': 2002,
        'Tricastin': 1981,
        'Dukovany': 1987,
        'Temelín': 2003,
        'Bohunice': 1985,
        'Mochovce': 2000,
        'Krško': 1983,
        'Ringhals': 1983,
        'Oskarshamn': 1985,
        'Forsmark': 1985,
        'Olkiluoto': 2019}

    Capacity = {
        'St laurent': 1830,
        'Gravelines': 5460,
        'Paluel': 5320,
        'Penly': 2660,
        'Nogent': 2620,
        'Golfech': 2620,
        'St alban': 2670,
        'Belleville': 2620,
        'Blayais': 3640,
        'Cruas': 3660,
        'Fessenheim': 0,
        'Flamanville': 4260,  # incl. new reactor being built
        'Dampierre': 3560,
        'Chinon': 3620,
        'Bugey': 3580,
        'Cattenom': 5200,
        'Chooz': 3000,
        'Civaux': 2990,
        'Tricastin': 3660,
        'Ringhals': 2166,
        'Oskarshamn': 1400,
        'Forsmark': 3269,
        'Dukovany': 1878,
        'Temelín': 2006,
        'Bohunice': 943,
        'Mochovce': 872,
        'Krško': 688,
        'Olkiluoto': 1600, #Capacity of the new reactor only
        'Brokdorf': 0,  # Set German capacitities to zero
        'Emsland': 0,
        'Grohnde': 0,
        'Gundremmingen': 0,
        'Isar': 0,
        'Neckarwestheim': 0,
        'Philippsburg': 0,
        'Biblis': 0,
        'Unterweser': 0,
        'Grafenrheinfeld': 0,
        'Kruemmel': 0}

    nuc = pd.DataFrame(ppl.loc[(ppl.Fueltype == "Nuclear"), ["Name", "DateIn", "Capacity"],])
    for name, year in iteritems(YearCommercial):
        name_match_b = nuc.Name.str.contains(name, case=False, na=False)
        if name_match_b.any():
            nuc.loc[name_match_b, "DateIn"] = year
        else:
            print("'{}' was not found in given DataFrame.".format(name))
        ppl.loc[nuc.index, "DateIn"] = nuc["DateIn"]

    for name, capacity in iteritems(Capacity):
        name_match_b = nuc.Name.str.contains(name, case=False, na=False)
        if name_match_b.any():
            nuc.loc[name_match_b, "Capacity"] = capacity
        else:
            print("'{}' was not found in given DataFrame.".format(name))
        ppl.loc[nuc.index, "Capacity"] = nuc["Capacity"]

    return ppl

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_powerplants')
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)
    countries = n.buses.country.unique()

    ppl = (pm.powerplants(from_url=True)
           .powerplant.fill_missing_decommyears()
           .powerplant.convert_country_to_alpha2()
           .query('Fueltype not in ["Solar", "Wind"] and Country in @countries')
           .replace({'Technology': {'Steam Turbine': 'OCGT'}})
            .assign(Fueltype=lambda df: (
                    df.Fueltype
                      .where(df.Fueltype != 'Natural Gas',
                             df.Technology.replace('Steam Turbine',
                                                   'OCGT').fillna('OCGT'))))
           .assign(DateIn=lambda df: (
                    df.DateIn
                      .where((df.Country == 'SE') & (df.Fueltype == 'Nuclear'),
                             df.DateIn.replace('2006', '1985'))))
           )

    #
    ppl_query = snakemake.config['electricity']['powerplants_filter']
    if isinstance(ppl_query, str):
        ppl.query(ppl_query, inplace=True)

    ppl = add_custom_powerplants(ppl) # add carriers from own powerplant files
    print(ppl)
    ppl = correct_nuclear_data(ppl)
    print(ppl)
    cntries_without_ppl = [c for c in countries if c not in ppl.Country.unique()]

    for c in countries:
        substation_i = n.buses.query('substation_lv and country == @c').index
        kdtree = KDTree(n.buses.loc[substation_i, ['x','y']].values)
        ppl_i = ppl.query('Country == @c').index

        tree_i = kdtree.query(ppl.loc[ppl_i, ['lon','lat']].values)[1]
        ppl.loc[ppl_i, 'bus'] = substation_i.append(pd.Index([np.nan]))[tree_i]

    if cntries_without_ppl:
        logging.warning(f"No powerplants known in: {', '.join(cntries_without_ppl)}")

    bus_null_b = ppl["bus"].isnull()
    if bus_null_b.any():
        logging.warning(f"Couldn't find close bus for {bus_null_b.sum()} powerplants")

    ppl.to_csv(snakemake.output[0])
