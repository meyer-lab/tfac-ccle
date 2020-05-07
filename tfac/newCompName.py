"""returns names of top three weighted proteins per component, each comp separated by 'Col Name'"""

def compProteins(comps):
    i=np.shape(comps)
    proteins, df_pnames = proteinNames()
    proteinNum, compNum = np.shape(comps[i[0]-1])
    compName = []
    proteinN=[]
    
    for x in range(0,compNum):
        compName.append('Col' + str(x+1))
        
    dfComps=pd.DataFrame(data=comps[i[0]-1], index=proteins, columns=compName)
    for y in range(0,compNum):
        proteinN.append(compName[y])
        rearranged = dfComps.sort_values(by=compName[y], ascending=False)
        rearrangedNames = list(rearranged.index.values)
        for z in range(0, 3):
            proteinN.append(rearrangedNames[z])

    return proteinN
