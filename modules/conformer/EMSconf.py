from typing import Type, List, Optional
import numpy as np
from EMS.EMS import EMS


class EMSconf(EMS):
    def __init__(self, EMSs: List[Type], mol_name="conf"):
        super().__init__(file)
        self.mol_name = mol_name
        self.num_confs = len(EMSs)
        self.EMSs = EMSs
        self.energy_array = np.zeros(self.num_confs, dtype=np.float64)
        self.pop_array = np.zeros(self.num_confs, dtype=np.float64)
        self.eliminated_mols = []
        self.averaged_shift = []
        self.averaged_coupling = []

    def calc_pops(self, temp: int = 298):
        """
        calculates the population distrubution of the different conformers used to instantiate the class, prerequisite step for boltzmann averaging
        and updates the EMS mol_properties attriputes with values

        :param temp: int, temperature to determine the spread of population
        :return: None
        """

        for c, EMS in enumerate(self.EMSs):
            self.energy_array[c] = EMS.energy

        kj_array = self.energy_array * 2625.5
        min_val = np.amin(kj_array)
        rel_array = kj_array - min_val
        exp_array = -(rel_array * 1000) / float(8.31 * temp)
        exp_array = np.exp(exp_array)
        sum_val = np.sum(exp_array)

        pop_array = exp_array / sum_val

        self.pop_array = pop_array

    def get_dist_array(self):

        for EMS in self.EMSs:
            num_atoms = len(EMS.type)
            dist_array = np.zeros((num_atoms, num_atoms), dtype=np.float64)
            for i in range(num_atoms):
                for j in range(num_atoms):
                    dist_array[i][j] = np.linalg.norm(EMS.xyz[i] - EMS.xyz[j])

            EMS.dist = dist_array

    def redundant_elimination(
        self,
        geom_threshold: float = 0.1,
        e_threshold: float = 0.1,
        redundant_atoms: Optional[list] = None,
        achiral: bool = False,
    ):

        if redundant_atoms != None:
            redundant_atoms = redundant_atoms.split(",")
            redundant_atoms = list(map(int, redundant_atoms))

            dist_arrays = np.zeros(len(self.EMSs[0].type), len(self.EMSs[0].type))
            for atoms in redundant_atoms:
                for i in range(len(self.EMSs)):
                    dist_arrays[i][int(atoms) - 1] = 0
                    for k in range(atoms):
                        dist_arrays[i][k][int(atoms) - 1] = 0

        with open("f{self.mol_name}_redundant_elimination_log.txt", "w") as f:
            for a, EMS_a in enumerate(self.EMSs):
                self.energy_array[a] = EMS_a.energy
                dist_array_a = EMS_a.dist

                for b, EMS_b in enumerate(self.EMSs):
                    self.energy_array[b] = EMS_b.energy
                    dist_array_b = EMS_b.dist
                    if a > b and not b in self.eliminated_mols:
                        diff = np.absolute(np.sum(dist_array_a - dist_array_b))
                        energy_diff = (
                            np.absolute(self.energy_array[a] - self.energy_array[b])
                            * 2625.5
                        )

                        if diff < geom_threshold:
                            if energy_diff < e_threshold:
                                self.eliminated_mols.append(b)
                                print(
                                    f"added mol {EMS_a.name} to eliminated_mol.txt due to geomtric similarity to {EMS_b.name}",
                                    file=f,
                                )
                            else:
                                if a - b == 1:
                                    print(
                                        f"energy difference between {EMS_a.id} & {EMS_b.id} detected but could be mirror images, please check manually",
                                        file=f,
                                    )
                                else:
                                    self.eliminated_mols.append(a)
                                    print(
                                        f"added mol {EMS_a.id} to eliminated_mol.txt due to geomtric similarity to {EMS_b.id} as mirror image has been found",
                                        file=f,
                                    )

                        else:
                            print(
                                f"geometry threshold is passed but not energy threshold, consider changing parameters after checking {EMS_a.id} & {EMS.b.id}",
                                file=f,
                            )
                            print(
                                f"{EMS_a.id} & {EMS_b.id} energy diff & geom diff = {energy_diff} kj/mol & {diff} Angstroms",
                                file=f,
                            )

        elim_list = list(set(self.eliminated_mols))
        removed_mols = []
        for mol in elim_list:
            removed_mols.append(mol)

        with open(f"{self.mol_name}_eliminated_mols.txt", "w") as f:
            for id in removed_mols:
                f.write(str(id) + "\n")

    def boltzmann_average(
        self, pair_props: list = ["coupling"], atom_props: list = ["shift"]
    ):
        atoms = len(self.EMSs[0].type)

        new_atom_dict = {}
        new_pair_dict = {}
        for prop in atom_props:
            new_atom_dict[prop] = np.zeros(atoms, dtype=np.float64)
        for prop in pair_props:
            new_pair_dict[prop] = np.zeros((atoms, atoms), dtype=np.float64)

        for i, EMS in enumerate(self.EMSs):
            for prop in atom_props:
                new_atom_dict[prop] += EMS.atom_properties[prop] * self.pop_array[i]
            for prop in pair_props:
                for atom in range(atoms):
                    new_pair_dict[prop][atom] += (
                        np.array(EMS.pair_properties[prop][atom]) * self.pop_array[i]
                    )

        self.averaged_shift
