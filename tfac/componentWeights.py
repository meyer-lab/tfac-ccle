def proteinNames():
    data=importLINCSprotein()
    data=data.drop(columns=['Treatment', 'Sample description', 'File', 'Time'],axis=1)
    proteinN=data.columns.values.tolist()
    return proteinN 
    
def componentWeights(decomps, proteinLabels):
    #takes in array of protein decomp'd components and list of proteinNames and picks out top 3 protein weights
    #components are by columns, proteins by rows
    i=np.shape(decomps)
    proteins=decomps[i[0]-1]
    proteinNum,compNum=np.shape(proteins)
    p_weights=np.zeros((3,compNum)) #stores indicies of 3 largest protein weights
    p_labels=[] #corresponding labels
    p_labels.append([])
    p_labels.append([])
    p_labels.append([])
    p_sorted=np.argsort(proteins,axis=0) #sorts by component from smallest to largest
    
    for y in range(0,compNum):
        for x in range(0,3):
            p_weights[x,y]=p_sorted[proteinNum-(x+1),y]
            ind=int(p_weights[x,y])
            p_labels[x].append(proteinLabels[ind])
        
    return p_weights, p_labels
