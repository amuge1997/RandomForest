
from SB_Data import arr_XData2,arr_LData2
import numpy as n



class RandomForest:
    def __init__(self,it_treeNum,it_selFeaNum,arr_X,arr_L):

        self.it_selFeaNum = it_selFeaNum
        self.arr_X = arr_X
        self.arr_L = arr_L

        self.ls_tree = [self.createTree() for i in range(it_treeNum)]

    def createTree(self):
        it_sampleNum = self.arr_X[0]
        it_featureNum = self.arr_X[1]
        arr_samplei = n.random.randint(0,it_sampleNum,it_sampleNum)
        arr_featurei = n.random.randint(0,it_featureNum,self.it_selFeaNum)
        dc = {
            'tree':CARTree(arr_X=self.arr_X[arr_samplei,arr_featurei],arr_L=self.arr_L[arr_samplei,:]),
            'sampleIndex':arr_samplei,
            'featureIndex':arr_featurei,
        }
        return dc

    def train(self):
        pass


if __name__ == '__main__':
    RF = RandomForest(3,3,arr_XData2,arr_LData2)





















