import ROOT
from root_numpy import root2array, tree2array
from numpy import save
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse

filename = sys.argv[1]
fileid = filename.split("_")[-1].split(".")[0]
process = filename.split("/")[-1].split("_")[0]

print filename
rfile = ROOT.TFile(filename)
intree = rfile.Get('events')

array = tree2array(intree, 
                   branches=['pt', 'eta', 'phi', 'e',
                             'vtxid', 'pdgid', 'puppi', 'genmet', 'hardfrac'],
                   selection='pt>0.',
                   start=0, stop=1000, step=1)

xarray = []
yarray = []
garray = []
genmet = []
puppiweights = []
counter = 0

for i in array:
    if counter%50 == 0:
        print str(counter) + " / " + str(len(array))
    genmet.append(i[7])
    npart = len(i[0])
    partsperevt = []
    psperevent = []
    yperevent = []
    xyzs = []
    for n in range(npart):
        tmpp4 = ROOT.TLorentzVector()
        tmpp4.SetPtEtaPhiE(i[0][n],i[1][n],i[2][n],i[3][n])
        xyz = [tmpp4.X(),tmpp4.Y(),tmpp4.Z()]
        xyzs.append(xyz)
        feature_arr = []
        for f in range(len(i)-3):
            feature_arr.append(i[f][n])
        partsperevt.append(feature_arr)
        yperevent.append([i[-1][n]])
        psperevent.append([i[6][n]])

    xyzs = np.array(xyzs)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(xyzs)
    spa = sparse.csr_matrix(np.array(nbrs.kneighbors_graph(xyzs).toarray()))
    nz = spa.nonzero()
    garray.append(np.array([nz[0],nz[1]]))
    xarray.append(partsperevt)
    yarray.append(yperevent)
    puppiweights.append(psperevent)

    counter += 1

xarray = np.array(xarray)
yarray = np.array(yarray)

save('%s_x_%s.npy'%(process,fileid), xarray)
save('%s_y_%s.npy'%(process,fileid), yarray)
save('%s_g_%s.npy'%(process,fileid), garray)
#save('%s_genmet_%s.npy'%(process,fileid), genmet)
#save('%s_pw_%s.npy'%(process,fileid), puppiweights)
