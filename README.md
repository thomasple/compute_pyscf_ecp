# Description
This program computes DFT energies and forces for molecular conformations using the PySCF library.
* DFT functional: wB97M-D3BJ
* basis set: aug-cc-pVTZ
* use effective core potentials (ccECP). It is designed to process datasets of molecular structures and output their corresponding energies and forces.
* processes input data in FeNNol dataset format.

This is the level of theory used to generate the training data for the FeNNix-Bio1 model.

# Units
* Distances in Angstrom.
* Energies in Hartree.
* Forces in Hartree/Angstrom.
* charges in elementary charge units.

# Usage
### Running DFT Calculations
Run example with:
```bash
OMP_NUM_THREADS=8 uv run main.py test_data/dataset_c4h6_finetune.pkl 
```

or with GPU support:
```bash
uv run --extra cuda main.py test_data/dataset_c4h6_finetune.pkl --gpu 0
```

### Reading Results
The output pickle file contains a sequence of dictionaries, each with the following keys:
- `species`: List of atomic numbers for each atom in the molecule.
- `coordinates`: Nx3 array of atomic coordinates.
- `total_charge`: Total charge of the molecule.
- `total_energy`: Computed total energy of the molecule.
- `total_energy_mask`: Boolean indicating if the energy computation was successful.
- `formation_energy`: Computed formation energy of the molecule.
- `formation_energy_mask`: Boolean indicating if the formation energy computation was successful.
- `forces`: Nx3 array of computed forces on each atom.
- `forces_mask`: Boolean indicating if the force computation was successful.
- `error`: Error message if the computation failed, otherwise None.
- `computation_time`: Time taken to compute the energy.

The list of frames can be read as follows:
```python
import pickle
with open("output_file.pkl", "rb") as f:
    data = []
    while True:
        try:
            entry = pickle.load(f)
            data.append(entry)
        except EOFError:
            break
```


### Sharding a Dataset
To split a dataset into shards for easier parallel processing, use the `make_shards.py` utility:
```bash
uv run make_shards.py test_data/dataset_c4h6_finetune.pkl 10
```