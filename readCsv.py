
def read_tree(filename):

    import pickle
    reader = open(filename, 'rU')
    return pickle.load(reader)


def test_tree( decesion_tree):
    root = decesion_tree.keys()[0]
    first_fea=root.split('<=')[0]
    values=root.split('<=')[1]
    
    print
    # fea_index = labels.index(first_fea)

tree=read_tree('decesiontree2')
print tree
test_tree(tree)
# import plotTree
#
# plotTree.createPlot(tree)