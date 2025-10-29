#!/usr/bin/env python3
import argparse
import numpy as np
import time
from functools import partial
import os
from pyscf import gto, dft
from pyscf.dft import rks
import glob
import pickle
import tqdm

from utils import PERIODIC_TABLE, PERIODIC_TABLE_REV_IDX, BOHR, read_pickle_dataset
from energy_references import neutral_atom_energies, isolated_atom_energies

RES_DIR = os.path.dirname(__file__) + "/resources"

print_flush = partial(print, flush=True)

species_avail = set(isolated_atom_energies.keys())
species_avail = np.array([PERIODIC_TABLE_REV_IDX[s] for s in species_avail])


def compute_entry(conformation, method="wb97m-d3bj",use_gpu=False):

    start_time = time.time()
    total_charge = round(conformation.get("total_charge", 0))
    coordinates = conformation["coordinates"]
    species = conformation["species"]
    if not np.all(np.isin(species, species_avail)):
        missing = set(species) - set(species_avail)
        print(f"Warning: Species {missing} not available in isolated atom energies. Formation energy will be wrong.")
    try:
        symbols = [PERIODIC_TABLE[s] for s in species]
        total_atomic_number = np.sum(species)
        multiplicity = 1 if (total_charge + total_atomic_number) % 2 == 0 else 2

        geom_str = f""

        for s, (x, y, z) in zip(symbols, coordinates):
            geom_str += f"{s} {x} {y} {z}\n"

        print_flush(geom_str)

        # mol = gto.M(atom=geom_str, basis={}, ecp={}, unit='angstrom',symmetry=False,charge=total_charge,spin=multiplicity-1)
        mol = gto.Mole()
        mol.atom = geom_str
        mol.unit = 'Ang'
        mol.charge = total_charge
        mol.spin= multiplicity-1
        mol.symmetry = False

        mol.basis={}
        mol.ecp={}
        symbols_set = set(symbols)
        for s in symbols_set:
        #   print(s)
          s_=str(s).strip()
          bas_file = glob.glob(f'{RES_DIR}/{s_}.aug*')[0]
        #   print(bas_file)
          with open(bas_file) as fb:
            bas_str = fb.read()
          mol.basis[s_]=gto.basis.parse(bas_str)
          ecp_file = glob.glob(f'{RES_DIR}/{s_}.ccECP*')[0]
        #   print(ecp_file)
          with open(ecp_file) as fb:
            ecp_str = fb.read()
          mol.ecp[s_] = gto.basis.parse_ecp(ecp_str)
           
        mol.build()

        mf = rks.RKS(mol).density_fit()
        # mf.with_df.auxbasis = {'default': 'aug-cc-pvtz-jkfit', 'I':'def2-universal-jkfit','Zn':'def2-universal-jkfit'}
        species_tz = ["H","B","C","N","O","F","Al","Si","P","S","Cl","Ga","Ge","As","Se","Br"]
        mf.with_df.auxbasis = {'default': 'def2-universal-jkfit', **{s: 'aug-cc-pvtz-jkfit' for s in species_tz}}
        mf.xc = method
        # mf.chkfile = title + '.chk'
        mf.conv_tol = 1e-10
        mf.direct_scf_tol = 1e-14
        mf.diis_space = 12
        mf.diis_start_cycle = 5
        mf.init_guess = ''

        mf.level_shift = 0.1
        mf.damp = 0.0
        mf.grids.level = 3
        mf.max_cycle = 300

        if use_gpu:
            mf = mf.to_gpu()

        e = mf.kernel()
        de = mf.nuc_grad_method().kernel()
        forces = -de/BOHR

        if use_gpu:
            mf.to_cpu()
        
        elapsed_time = time.time() - start_time

        eref = np.sum(neutral_atom_energies[species])
        eform = e - eref

        output = dict(
            species=species,
            coordinates=coordinates,
            total_charge=total_charge,
            total_energy=e,
            total_energy_mask=True,
            formation_energy=eform,
            formation_energy_mask=True,
            forces=forces,
            forces_mask=np.ones(coordinates.shape[0], dtype=bool),
            error=None,
            computation_time=elapsed_time,
        )
        

    except Exception as ex:
        print_flush(f"Error computing entry: {ex}")
        elapsed_time = time.time() - start_time
        output_data = dict(
            species=species,
            coordinates=coordinates,
            total_charge=total_charge,
            total_energy=0.,
            total_energy_mask=False,
            formation_energy=0.,
            formation_energy_mask=False,
            forces=np.zeros_like(coordinates),
            forces_mask=np.zeros(coordinates.shape[0], dtype=bool),
            error=str(ex),
            computation_time=elapsed_time,
        )
        output = {**conformation, **output_data}

    return output


def main():
    parser = argparse.ArgumentParser(description="compute wB97m-D3BJ/aug-cc-pvtz/ECP using pyscf")
    parser.add_argument(
        "pkl_file", type=str, help="Input pickle file with conformations (FeNNol format)"
    )
    parser.add_argument(
       "--gpu", type=int, default=-1, help="GPU id to use. If negative, use CPU"
    )
    parser.add_argument("-o","--output_file", type=str, help="Output pickle file to store results")

    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    tmp_dir = os.environ.get("PYSCF_TMPDIR", "/tmp/pyscf")

    print("tmp dir=",tmp_dir)

    assert os.path.exists(args.pkl_file), f"Input file {args.pkl_file} does not exist"
    assert args.pkl_file.endswith(".pkl"), "Input file must be a .pkl file"
    output_file = args.output_file
    if output_file is None:
        output_file = args.pkl_file.replace(".pkl", ".wb97m-d3bj_ecp.pkl")
    assert not os.path.exists(output_file), f"Output file {output_file} already exists"

    print(f"Loading conformations from {args.pkl_file}...")
    conformations = read_pickle_dataset(args.pkl_file)
    print(f"Loaded {len(conformations)} conformations.")
    
    
    print(f"Computing energies and forces with wB97m-D3BJ/aug-cc-pvtz/ECP...")
    with open(output_file, "wb") as f:
        for i, conf in enumerate(tqdm.tqdm(conformations)):
            # print_flush(f"Computing conformation {i+1}/{len(conformations)}")
            res = compute_entry(conf, use_gpu=(args.gpu >= 0))
            pickle.dump(res, f)
    print(f"Results saved to {output_file}.")

if __name__ == "__main__":
    main()
