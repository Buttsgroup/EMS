import pandas as pd
from tqdm import tqdm
from EMS.utils.periodic_table import Get_periodic_table

def make_atoms_df(ems_list, write=False, format="pickle"):
    p_table = Get_periodic_table()

    # construct dataframes
    # atoms has: molecule_name, atom, labeled atom,
    molecule_name = []  # molecule name
    atom_index = []  # atom index
    typestr = []  # atom type (string)
    typeint = []  # atom type (integer)
    x = []  # x coordinate
    y = []  # y coordinate
    z = []  # z coordinate
    conns = []
    atom_props = []
    for propname in ems_list[0].atom_properties.keys():
        atom_props.append([])

    pbar = tqdm(ems_list, desc="Constructing atom dictionary", leave=False)

    m = -1
    for ems in pbar:
        m += 1
        # Add atom values to lists
        for t, type in enumerate(ems.type):
            molecule_name.append(ems.id)
            atom_index.append(t)
            typestr.append(p_table[type])
            typeint.append(type)
            x.append(ems.xyz[t][0])
            y.append(ems.xyz[t][1])
            z.append(ems.xyz[t][2])
            conns.append(ems.conn[t])
            smiles = ems.mol_properties["SMILES"]
            for p, prop in enumerate(ems.atom_properties.keys()):
                atom_props[p].append(ems.atom_properties[prop][t])

    # Construct dataframe
    atoms = {
        "molecule_name": molecule_name,
        "atom_index": atom_index,
        "typestr": typestr,
        "typeint": typeint,
        "x": x,
        "y": y,
        "z": z,
        "conn": conns,
        "SMILES": smiles,
    }
    for p, propname in enumerate(ems.atom_properties.keys()):
        atoms[propname] = atom_props[p]

    atoms = pd.DataFrame(atoms)

    pbar.close()

    atoms.astype(
        {
            "molecule_name": "category",
            "atom_index": "Int16",
            "typestr": "category",
            "typeint": "Int8",
            "x": "Float32",
            "y": "Float32",
            "z": "Float32",
            "SMILES": "category",
        }
    )

    if write:
        if format == "csv":
            atoms.to_csv(f"{write}/atoms.csv")
        elif format == "pickle":
            atoms.to_pickle(f"{write}/atoms.pkl")
        elif format == "parquet":
            atoms.to_parquet(f"{write}/atoms.parquet")

    else:
        return atoms


def make_pairs_df(ems_list, write=False, max_pathlen=6):
    # construct dataframe for pairs in molecule
    molecule_name = []  # molecule name
    atom_index_0 = []  # atom index for atom 1
    atom_index_1 = []  # atom index for atom 2
    dist = []  # distance between atoms
    path_len = []  # number of pairs between atoms (shortest path)
    pair_props = []
    for propname in ems_list[0].pair_properties.keys():
        pair_props.append([])

    pbar = tqdm(ems_list, desc="Constructing pairs dictionary", leave=False)

    m = -1
    for ems in pbar:
        m += 1

        for t, type in enumerate(ems.type):
            for t2, type2 in enumerate(ems.type):
                # Add pair values to lists
                if ems.path_topology[t][t2] > max_pathlen:
                    continue
                molecule_name.append(ems.id)
                atom_index_0.append(t)
                atom_index_1.append(t2)
                dist.append(ems.path_distance[t][t2])
                path_len.append(int(ems.path_topology[t][t2]))
                for p, prop in enumerate(ems.pair_properties.keys()):
                    pair_props[p].append(ems.pair_properties[prop][t][t2])

    # Construct dataframe
    pairs = {
        "molecule_name": molecule_name,
        "atom_index_0": atom_index_0,
        "atom_index_1": atom_index_1,
        "dist": dist,
        "path_len": path_len,
    }
    for p, propname in enumerate(ems.pair_properties.keys()):
        pairs[propname] = pair_props[p]

    pairs = pd.DataFrame(pairs)

    pbar.close()

    if write:
        pairs.to_pickle(f"{write}/pairs.pkl")
    else:
        return pairs