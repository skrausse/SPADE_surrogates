# SPADE_surrogates
Repository containing code necessary to reproduce the results of Stella, A., Bouss, P., Palm, G., &amp; Grün, S. (2021). Generating surrogates for significance estimation of spatio-temporal spike patterns

To initialize the submodule `multielectrode_grasp` in data/ after cloning the repository run:
`git clone --recurse-submodules git@github.com:INM-6/SPADE_surrogates.git`
If you already cloned the project and forgot, you can run `git submodule update --init`.

In order to download the data, please, go into the multielectrode_grasp folder and first do `git checkout 24cd5caee3`
loading the last tested commit (by us). Then inside datasets, run `gin login`. If you don't 
have a gin-account or client, refer to the README in datasets. Then you need to run:
`gin get-content i140703-001-03.nev l101210-001-02.nev i140703-001.odml l101210-001.odml`. This will download the
results of the spike-sorting (.nev) and metadata files (.odml). 
