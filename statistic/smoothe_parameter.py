#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
from util import constants
import scipy.special as special
import os
import numpy
import random
import gc


class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = ((special.digamma(tries - success + beta) - special.digamma(beta))).sum()
        sumfenmu = ((special.digamma(tries+alpha+beta) - special.digamma(alpha+beta))).sum()

        # for i in range(len(tries)):
        #     sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
        #     sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
        #     sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))
        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation'''
        ctr_list = []
        # var = 0.0
        mean = (success / tries).mean()
        if len(tries) == 1:
            var = 0
        else:
            var = (success/tries).var()
        # for i in range(len(tries)):
        #     ctr_list.append(float(success[i])/tries[i])
        # mean = sum(ctr_list)/len(ctr_list)
        # for ctr in ctr_list:
        #     var += pow(ctr-mean, 2)

        return mean, var


def extra_rate_active(train, key):
    # print '开始生rate,active'
    key2 = []
    key_string = ''
    for e in key:
        key2.append(e)
        key_string = key_string + '_' + e
    key2.append('label')
    fea = train[key2]
    count = 'count' + key_string
    fea[count] = 1
    fea = fea.groupby(key).agg('sum').reset_index()
    # 平滑
    hyper = HyperParam(1, 1)
    I = fea[count]
    C = fea['label']
    # hyper.update_from_data_by_FPI(I, C, 1000, 0.00001)#000
    hyper.update_from_data_by_moment(I, C)
    print key_string, hyper.alpha, hyper.beta

train = pd.read_csv(constants.custom_path+'/train_with_ad_info.csv')
train = train[['label', 'appID']]
gc.collect()
key = list(['appID'])
extra_rate_active(train, key)