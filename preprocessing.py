import numpy as np
import pickle as pkl
import sys, sparse
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


data = sys.argv[1]
use_AROMATIC = False

def isProperty(adj):
    nodes = 9 # Number of nodes
    adj_lt = np.tril(adj) # Lower Triangular form
    edges = np.sum(adj_lt) # Number of edges
    avg_degree = 2*edges/nodes

    A3 = adj@adj@adj
    tri = (1.0/6)*np.trace(A3)

    if 2 <= avg_degree <= 3:
        if tri >= 2:
            return 1
    return 0


def to_onehot(val, cat):

    vec = np.zeros(len(cat))
    for i, c in enumerate(cat):
        if val == c: vec[i] = 1

    if np.sum(vec) == 0: print('* exception: missing category', val)
    assert np.sum(vec) == 1

    return vec

def atomFeatures(a):

    v1 = to_onehot(a.GetFormalCharge(), [-1, 1, 0])[:2]
    v2 = to_onehot(a.GetNumExplicitHs(), [1, 2, 3, 0])[:3]    
    v3 = to_onehot(a.GetSymbol(), atom_list)
    
    return np.concatenate([v1, v2, v3], axis=0)

def bondFeatures(bonds):

    e1 = np.zeros(dim_edge)
    if len(bonds)==1:
        e1 = to_onehot(str(bonds[0].GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])[:dim_edge]

    return np.array(e1)

def MolFromGraphs(node_list, adjacency_matrix):

    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(int(node_list[i]))
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()

    return mol


if data=='QM9':
    data_size=100000
    n_max=9
    atom_list=['C','N','O','F']

elif data=='ZINC':
    data_size=100000
    n_max=38
    atom_list=['C','N','O','F','P','S','Cl','Br','I']
    
elif data=='data':
    data_size=2000
    n_max=9
    atom_list=['C']

dim_node = 5 + len(atom_list)
dim_edge = 3
if use_AROMATIC:
    dim_edge = dim_edge + 1
    
graphs = pkl.load(open('./'+data+'_smi.pkl','rb'))

DV = []
DE = [] 
DY = []
Dsmi = []
for i in range(graphs.shape[0]):
    mol = MolFromGraphs(np.array([6, 6, 6, 6, 6, 6, 6, 6, 6], dtype=np.int8), graphs[i])

    #if use_AROMATIC == False: Chem.Kekulize(mol)
    n_atom = mol.GetNumHeavyAtoms()
        
    # node DV
    node = np.zeros((n_max, dim_node), dtype=np.int8)
    for j in range(n_atom):
        atom = mol.GetAtomWithIdx(j)
        node[j, :]=atomFeatures(atom)
    
    # edge DE
    edge = np.zeros((n_max, n_max, dim_edge), dtype=np.int8)
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            molpath = Chem.GetShortestPath(mol, j, k)
            bonds = [mol.GetBondBetweenAtoms(molpath[bid], molpath[bid + 1]) for bid in range(len(molpath) - 1)]
            edge[j, k, :] = bondFeatures(bonds)
            edge[k, j, :] = edge[j, k, :]

    # property DY
    property = [Descriptors.ExactMolWt(mol), isProperty(graphs[i])]

    print (property)

    # append
    DV.append(node)
    DE.append(edge)
    DY.append(property)

    if use_AROMATIC: Dsmi.append(Chem.MolToSmiles(mol))
    else: Dsmi.append(Chem.MolToSmiles(mol, kekuleSmiles=True))

    if i % 1000 == 0:
        print(i, flush=True)

    if len(DV) == data_size: break

# np array    
DV = np.asarray(DV, dtype=np.int8)
DE = np.asarray(DE, dtype=np.int8)
DY = np.asarray(DY)
Dsmi = np.asarray(Dsmi)

# compression
DV = sparse.COO.from_numpy(DV)
DE = sparse.COO.from_numpy(DE)

print (DY)

# save
with open(data+'_graph.pkl','wb') as fw:
    pkl.dump([DV, DE, DY, Dsmi], fw)
