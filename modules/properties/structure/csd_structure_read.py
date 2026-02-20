from ccdc.molecule import Molecule

def cif_to_csdmol(file):
    """
    This function reads in a cif and returns a csdmol object of the heaviest component in the cif - removes any salts/solvents 
    
    :param file: Description
    :param mol_id: Description
    """
    with open(file, "r") as f:
        cif_string = f.read()

    mols = Molecule.from_string(cif_string, format="cif")
    heaviest_component = mols.heaviest_component
    csdmol = heaviest_component


    return csdmol
