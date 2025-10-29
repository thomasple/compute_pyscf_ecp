import pickle


BOHR = 0.52917721067

PERIODIC_TABLE_STR = """
H                                                                                                                           He
Li  Be                                                                                                  B   C   N   O   F   Ne
Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
"""

PERIODIC_TABLE = ["Dummy"] + PERIODIC_TABLE_STR.strip().split()

PERIODIC_TABLE_REV_IDX = {s: i for i, s in enumerate(PERIODIC_TABLE)}

def read_pickle_frames(pkl_file):
    data = []
    with open(pkl_file, "rb") as f:
        while True:
            try:
                frame = pickle.load(f)
                data.append(frame)
            except EOFError:
                break
    return data

def read_pickle_dataset(pkl_file):
    with open(pkl_file, "rb") as f:
        dataset = pickle.load(f)
    if not isinstance(dataset, list):
        assert isinstance(dataset, dict), "Input pickle file must contain a list or dict"
        dataset = dataset["training"]+dataset["validation"]
    assert isinstance(dataset, list), "Dataset must be a list of conformations"
    return dataset

def write_pickle_dataset(pkl_file, dataset):
    with open(pkl_file, "wb") as f:
        pickle.dump(dataset, f)