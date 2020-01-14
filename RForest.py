from RandomForest.Tree.CARTree import CARTree
from SB_Data import arr_XData2,arr_LData2,ls_XData2FeatureName
import numpy as n
import random



class RandomForest:
    def __init__(self,arr_X,arr_L,it_treeNum,it_selFeaNum=None,ls_feaName=None):
        self.it_treeNum = it_treeNum                    # 树的数量,树太少不稳定

        self.arr_X = arr_X
        self.it_sampleNum = self.arr_X.shape[0]
        self.it_minSample = int(self.it_sampleNum / 10)
        self.it_featureNum = self.arr_X.shape[1]

        self.arr_L = arr_L
        self.it_clfNum = arr_L.shape[1]
        if it_selFeaNum is None:
            self.it_selFeaNum = int(self.it_featureNum/3)       # 如果不指定特征数量则使用 1/3 的特征
        else:
            self.it_selFeaNum = it_selFeaNum

        if self.it_selFeaNum > self.it_featureNum:
            raise Exception('RandomForest.init() 特征选择个数超过了样本特征个数!')

        self.ls_tree = [self.createTree(ls_feaName) for i in range(self.it_treeNum)]

    def createTree(self,ls_feaName):
        arr_samplei = n.random.randint(0,self.it_sampleNum,self.it_sampleNum)     # 可重复
        arr_featurei = n.array(random.sample([i for i in range(self.it_featureNum)], self.it_selFeaNum)) # 不可重复
        dc_tree = {
            'tree':CARTree(
                arr_X=self.arr_X[n.ix_(arr_samplei,arr_featurei)],
                arr_L=self.arr_L[arr_samplei,:],
                it_minSample=self.it_minSample,
            ),
            'sampleIndex':arr_samplei,
            'featureIndex':arr_featurei,
            'oobRate':0.0,
        }
        fl_oobRate = self.calTreeOOB(dc_tree=dc_tree)
        dc_tree['oobRate'] = fl_oobRate
        if ls_feaName is not None:
            dc_tree['featureName'] = n.array(ls_feaName)[arr_featurei]
        return dc_tree

    def calTreeOOB(self,dc_tree):
        arr_X = self.arr_X
        arr_L = self.arr_L
        it_sampleNum = self.it_sampleNum
        it_featureNum = self.it_featureNum
        it_clfNum = self.it_clfNum

        arr_sampleAlli = n.arange(0,it_sampleNum,1)

        ins_tree = dc_tree['tree']
        arr_samplei = dc_tree['sampleIndex']
        arr_featurei = dc_tree['featureIndex']
        arr_diff = n.setdiff1d(arr_sampleAlli,arr_samplei)

        it_oobTrue = 0      # oob正确样本
        it_oobSampleNum = arr_diff.shape[0]     # oob总样本

        for it_i in arr_diff:
            arr_oneX = arr_X[it_i].reshape(-1, it_featureNum)[:,arr_featurei]
            arr_oneL = arr_L[it_i].reshape(-1, it_clfNum)
            arr_treePre = ins_tree.predict(arr_oneX)
            arr_treeFinPre = n.zeros((1,it_clfNum))         # 转换成只包含0,1的预测数组
            arr_treeFinPre[0,n.argmax(arr_treePre)] = 1.0

            if n.array_equal(arr_oneL,arr_treeFinPre):
                it_oobTrue += 1
        fl_oobRate = it_oobTrue / it_oobSampleNum
        return fl_oobRate


    def predict(self,arr_X):
        arr_treePre = n.zeros((1,self.it_clfNum))
        for dc_tree in self.ls_tree:
            arr_featurei = dc_tree['featureIndex']
            ins_tree = dc_tree['tree']
            fl_oobRate = dc_tree['oobRate']
            arr_feaX = arr_X[:,arr_featurei]
            p = ins_tree.predict(arr_feaX)
            arr_treePre += p * fl_oobRate       # 基于袋外误差率的加权平均进行预测

        arr_treeFinPre = arr_treePre/(n.sum(arr_treePre))
        return arr_treeFinPre

if __name__ == '__main__':
    RF = RandomForest(arr_XData2,arr_LData2,it_treeNum=20,it_selFeaNum=5,ls_feaName=ls_XData2FeatureName)

    dc = {
        'face': 2,  # 0
        'rf':   1,  # 1
        'tui':  2,  # 2
        'high': 2,  # 3
        'fm':   1,  # 4
        'age':  0,  # 5
    }
    ls_ = [v for k, v in dc.items()]
    X = n.array([ls_])

    pre = RF.predict(X)
    print(pre)

















