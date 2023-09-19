# nautilos-T8.5-sst

This repository contains a Python 3 implementation of the Mesoscale Events Classifier (MEC) algorithm [\[1\]](#applsci-mec), which has been developed as part of the activities of Task 8.5 of the [NAUTILOS](https://www.nautilos-h2020.eu) project. This algorithm uses Sea Surface Temperature (SST) data coming from satellite missions to detect and classify patterns associated with "mesoscale events" in an upwelling ecosystem.

This work is part of a project that has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 101000825 (NAUTILOS).

## Files Description

### spaghetti_plot.py

This file contains:

- the definitions of two custom Python classes, namely `SpaghettiData` and `SpaghettiPlot`, used to organise SST data (see [\[2\]](#spaghetticlasses) for more details);
- the core scripts of the MEC algorithm;
- a series of tools to visualise the results of the single steps and the output of the algorithm.

### sst_utils.py

This file contains scripts that extract the SST information from satellite products (in the form of NetCDF files) and prepare it for the subsequent steps of the algorithm.

### mec_geodata.py

This file contains two utility scripts that produce some arrays with geographical information used by MEC.

## References

<a id="applsci-mec">\[1\]</a> Gabriele Pieri, João Janeiro, Flávio Martins, Oscar Papini and Marco Reggiannini. "MEC: A Mesoscale Events Classifier for Oceanographic Imagery". In: _Applied Sciences_ 13.3, 1565 (2023). DOI: [10.3390/app13031565](https://doi.org/10.3390/app13031565).

<a id="spaghetticlasses">\[2\]</a> Oscar Papini. `SpaghettiData` _and_ `SpaghettiPlot`: _two Python classes for analysing and visualising SST trends_. ISTI Technical Reports 2022/001. ISTI-CNR, 2022. DOI: [10.32079/ISTI-TR-2022/001](https://doi.org/10.32079/ISTI-TR-2022/001).
