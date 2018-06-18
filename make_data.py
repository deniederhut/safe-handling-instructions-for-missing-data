#!/usr/env python
#! -*- encoding: utf-8 -*-

import itertools
import logging
import sqlalchemy as sqla
import time

from impyute.imputations.cs import em
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import LinearSVR


class ListwiseDel:

    def fit(self, x):
        locations = np.where((~np.isnan(x)).all(axis=-1))
        self.rows = locations[0]
        return self

    def transform(self, x):
        x = np.array(x)
        return x[self.rows, :]

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class EMImputer:

    def __init__(self):
        pass

    def fit_transform(self, x):
        x = np.array(x)
        return em(x)

class SoftImputer:

    def __init__(self):
        self.imputer = SoftImpute()

    def fit_transform(self, x):
        x = np.array(x)
        return self.imputer.complete(x)


class Main:

    def __init__(self, *args, outfile, sizes, trials, **kwargs):
        self.sizes = sizes
        self.trials = trials
        self.strats = [listwise_del, mean_imputer, median_imputer, em_imputer]
        self.generations = [linear_data, quadratic_data, wave_data]
        self.cats = [nothing, mar, mcar, mnar]
        self.fracs = [0.9, 0.8, 0.7, 0.6, 0.5]
        self.eweights = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.cvstrength = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.tasks = [LinearRegression, Lasso, Ridge, LinearSVR]
        self.outfile = outfile

        columns = ['size', 'category', 'task', 'strat', 'generation', 'frac', 'eweight', 'cvstrength',
                   'trial', 'score', 'coef_x', 'coef_y']
        self.results = {}
        for name in columns:
            self.results[name] = []

    def get_coefs(self, train, test, size, gen, cat, strat, task, frac, eweight, cvstrength, trial):
        model = task()
        model.fit(train[:, :2], train[:, -1:])
        preds = model.predict(test[:, :2])
        score = mean_squared_error(test[:, -1:], preds)
        if model.coef_.ndim == 2:
            x, y = model.coef_[0]
        else:
            x, y = model.coef_
        self.results['size'].append(size)
        self.results['category'].append(cat.__name__)
        self.results['strat'].append(strat.__name__)
        self.results['generation'].append(gen.__name__)
        self.results['task'].append(task.__name__)
        self.results['frac'].append(frac)
        self.results['eweight'].append(eweight)
        self.results['cvstrength'].append(cvstrength)
        self.results['trial'].append(trial)
        self.results['score'].append(score)
        self.results['coef_x'].append(x)
        self.results['coef_y'].append(y)
        logging.debug(str(self.results))

    def run(self):
        for gen, size, eweight, trial, cvstrength in itertools.product(self.generations, self.sizes, self.eweights, range(self.trials), self.cvstrength):
            x_train, x_test, y_train, y_test = gen(size, eweight, cvstrength)
            train = np.concatenate((x_train, y_train), axis=-1)
            test = np.concatenate((x_test, y_test), axis=-1)
            for cat, strat, task, frac in itertools.product(self.cats, self.strats, self.tasks, self.fracs):
                logging.info('Starting job::{}'.format((size, cat, task, strat)))
                missing = apply_missingness(train, cat)
                filled = apply_strategy(missing, strat)
                self.get_coefs(filled, test, size, gen, cat, strat, task, frac, eweight, cvstrength, trial)
        df = pd.DataFrame(self.results)
        df.to_csv(self.outfile+'.csv', index=False)
        df.to_sql(self.outfile+'.sqlite', table='results', if_exists='replace', index=False)

def _base_data(size, cvstrength):
    x = np.random.randn(size).reshape(-1, 1)
    y = np.random.randn(size).reshape(-1, 1)
    c_1 = (cvstrength * x + (1-cvstrength)
           * np.random.randn(size).reshape(-1, 1))
    c_2 = (cvstrength * x + (1-cvstrength)
           * np.random.randn(size).reshape(-1, 1))
    c_3 = (cvstrength * x + (1-cvstrength)
           * np.random.randn(size).reshape(-1, 1))
    data = np.concatenate((x, y, c_1, c_2, c_3), axis=-1)
    return data

def linear_data(size, error_weight=0.1, cvstrength=0.0):
    data = _base_data(size, cvstrength)
    x = data[:, 0]
    y = data[:, 1]
    t = (2*x + y + error_weight*np.random.random(size)).reshape(-1, 1)
    return train_test_split(data, t, train_size=0.8)

def quadratic_data(size, error_weight=0.1, cvstrength=0.0):
    data = _base_data(size, cvstrength)
    x = data[:, 0]
    y = data[:, 1]
    t = (x**2 - y + error_weight*np.random.random(size)).reshape(-1, 1)
    return train_test_split(data, t, train_size=0.8)

def wave_data(size, error_weight=0.1, cvstrength=0.0):
    data = _base_data(size, cvstrength)
    x = data[:, 0]
    y = data[:, 1]
    t = (2*np.sin(x) - y + error_weight*np.random.random(size)).reshape(-1, 1)
    return train_test_split(data, t, train_size=0.8)

def apply_strategy(data, strategy):
    instance = strategy()
    return instance.fit_transform(data)

def apply_missingness(data, category):
    data = np.array(data)
    return category(data)

def nothing(x):
    return x

def mcar(x, frac=0.1):
    x = np.array(x)
    n_choices = int(x.shape[0] * frac)
    rows = np.random.choice(range(x.shape[0]), n_choices)
    np.ravel(x)[rows] = np.nan
    return x

def mnar(x, frac=0.1):
    x = np.array(x)
    amount = int(x.shape[0] * frac)
    n_smallest = np.argsort(x[:, 1])[:amount]
    x[n_smallest, 0] = np.nan
    return x

def mar(x, frac=0.1):
    x = np.array(x)
    amount = int(x.shape[0] * frac)
    n_smallest = np.argsort(x[:, -1])[:amount]
    x[n_smallest, 0] = np.nan
    return x

def em_imputer():
    return EMImputer()

def listwise_del():
    return ListwiseDel()

def median_imputer():
    return Imputer(strategy='median')

def mean_imputer():
    return Imputer(strategy='mean')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sizes', nargs='+', type=int, default=100,
                        dest='sizes')
    parser.add_argument('-f', '--filename', dest='outfile',
                        help='Output basename for csv/sqlite')
    parser.add_argument('-t', '--trials', type=int, default=1,
                        dest='trials', help="Number of trials to run")
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')
    args = parser.parse_args()

    if args.outfile is None:
        args.outfile = 'make_data' + time.asctime().replace(' ', '-')

    logger = logging.getLogger()
    logger.setLevel(10)
    s_handle = logging.StreamHandler()
    if args.verbose:
        s_handle.setLevel(logging.DEBUG)
    else:
        s_handle.setLevel(logging.INFO)
    s_handle.setFormatter(logging.Formatter(
        "%(asctime)s::%(levelname)s::%(msg)s"
    ))
    logger.addHandler(s_handle)

    logging.debug("Argparser found {!s}".format(args))

    np.random.seed(42)

    main = Main(**vars(args))
    main.run()
