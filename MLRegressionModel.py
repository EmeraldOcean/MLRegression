import os
import joblib
import json

from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor


class MLRegressionConfig:
    def __init__(self, filePath, modelName):
        # sklearn.linear_model : https://scikit-learn.org/1.5/api/sklearn.linear_model.html
        self.filePath = filePath
        self.modelName = modelName

        with open(self.filePath, 'r') as f:
            self.configFile = json.load(f)

        if self.modelName == 'LinearRegression':
            config = self.configFile['LinearRegression']
            self.fit_intercept = config['fit_intercept']
            self.copy_X = config['copy_X']
            self.n_jobs = config['n_jobs']
            self.positive = config['positive']

        elif self.modelName == 'LassoRegression':
            config = self.configFile['LassoRegression']
            self.alpha = config['alpha']
            self.fit_intercept = config['fit_intercept']
            self.precompute = config['precompute']
            self.copy_X = config['copy_X']
            self.max_iter = config['max_iter']
            self.tol = config['tol']
            self.warm_start = config['warm_start']
            self.positive = config['positive']
            self.random_state = config['random_state']
            self.selection = config['selection']

        elif self.modelName == 'RidgeRegression':
            config = self.configFile['RidgeRegression']
            self.alpha = config['alpha']
            self.fit_intercept = config['fit_intercept']
            self.copy_X = config['copy_X']
            self.max_iter = config['max_iter']
            self.tol = config['tol']
            self.solver = config['solver']
            self.positive = config['positive']
            self.random_state = config['random_state']

        elif self.modelName == 'ElasticNet':
            config = self.configFile['ElasticNet']
            self.alpha = config['alpha']
            self.l1_ratio = config['l1_ratio']
            self.fit_intercept = config['fit_intercept']
            self.precompute = config['precompute']
            self.max_iter = config['max_iter']
            self.copy_X = config['copy_X']
            self.tol = config['tol']
            self.warm_start = config['warm_start']
            self.positive = config['positive']
            self.random_state = config['random_state']
            self.selection = config['selection']

        elif self.modelName == 'MultiTaskElasticNet':
            config = self.configFile['MultitaskElasticNet']
            self.alpha = config['alpha']
            self.l1_ratio = config['l1_ratio']
            self.fit_intercept = config['fit_intercept']
            self.precompute = config['precompute']
            self.max_iter = config['max_iter']
            self.copy_X = config['copy_X']
            self.tol = config['tol']
            self.warm_start = config['warm_start']
            self.positive = config['positive']
            self.random_state = config['random_state']
            self.selection = config['selection']

        elif self.modelName == 'PolynomialRegression':
            config = self.configFile['PolynomialRegression']
            self.degree = config['degree']
            self.interaction_only = config['interaction_only']
            self.include_bias = config['include_bias']
            self.order = config['order']

        elif self.modelName == 'RandomForestRegression':
            config = self.configFile['RandomForestRegression']
            self.n_estimators = config['n_estimators']
            self.criterion = config['criterion']
            self.max_depth = config['max_depth']
            self.min_samples_split = config['min_samples_split']
            self.min_samples_leaf = config['min_samples_leaf']
            self.min_weight_fraction_leaf = config['min_weight_fraction_leaf']
            self.max_features = config['max_features']
            self.max_leaf_nodes = config['max_leaf_nodes']
            self.min_impurity_decrease = config['min_impurity_decrease']
            self.bootstrap = config['bootstrap']
            self.oob_score = config['oob_score']
            self.n_jobs = config['n_jobs']
            self.random_state = config['random_state']
            self.verbose = config['verbose']
            self.warm_start = config['warm_start']
            self.ccp_alpha = config['ccp_alpha']
            self.max_samples = config['max_samples']
            self.monotonic_cst = config['monotonic_cst']

        elif self.modelName == "KNNRegresssion":
            config = self.configFile['KNeighborsRegresssion']
            self.n_neighbors = config['n_neighbors']
            self.weights = config['weights']
            self.algorithms = config['algorithms']
            self.leaf_size = config['leaf_size']
            self.p = config['p']
            self.metric = config['metric']
            self.metric_params = config['metric_params']
            self.n_jobs = config['n_jobs']

        elif self.modelName == 'XGBoost':
            # XGBoost : https://xgboost.readthedocs.io/en/stable/parameter.html
            config = self.configFile['XGBoost']
            self.booster = config['booster']
            self.verbosity = config['verbosity']
            self.validate_parameters = config['validate_parameters']
            self.disable_default_eval_metric = config['disable_default_eval_metric']

            self.eta = config['eta']
            self.gamma = config['gamma']
            self.max_depth = config['max_depth']
            self.min_child_weight = config['min_child_weight']
            self.max_delta_step = config['max_delta_step']
            self.subsample = config['subsample']
            self.sampling_method = config['sampling_method']
            self.colsample_bytree = config['colsample_bytree']
            self.colsample_bylevel = config['colsample_bylevel']
            self.colsample_bynode = config['colsample_bynode']

            self.alpha = config['alpha']
            self.tree_method = config['tree_method']
            self.scale_pos_weight = config['scale_pos_weight']
            self.refresh_leaf = config['refresh_leaf']
            self.process_type = config['process_type']
            self.grow_policy = config['grow_policy']
            self.max_leaves = config['max_leaves']
            self.max_bin = config['max_bin']
            self.num_parallel_tree = config['num_parallel_tree']
            self.monotone_constraints = config['monotone_constraints']
            self.interaction_constraints = config['interaction_constraints']
            self.multi_strategy = config['multi_strategy']

        else:
            raise ValueError("Unknown modelName. Edd to class")

class _LinearRegression(LinearRegression, MLRegressionConfig):
    def __init__(self, filePath):
        self.filePath = filePath
        MLRegressionConfig.__init__(self, self.filePath, "LinearRegression")
        LinearRegression.__init__(self,
                                  fit_intercept=self.fit_intercept,
                                  copy_X=self.copy_X,
                                  n_jobs=self.n_jobs,
                                  positive=self.positive)

class _LassoRegression(Lasso, MLRegressionConfig):
    def __init__(self, filePath):
        self.filePath = filePath
        MLRegressionConfig.__init__(self, self.filePath, "LassoRegression")
        Lasso.__init__(self,
                       alpha=self.alpha,
                       fit_intercept=self.fit_intercept,
                       precompute=self.precompute,
                       copy_X=self.copy_X,
                       max_iter=self.max_iter,
                       tol=self.tol,
                       warm_start=self.warm_start,
                       positive=self.positive,
                       random_state=self.random_state,
                       selection=self.selection)

class _RidgeRegression(Ridge, MLRegressionConfig):
    def __init__(self, filePath):
        self.filePath = filePath
        MLRegressionConfig.__init__(self, self.filePath, "RidgeRegression")
        Ridge.__init__(self,
                       alpha=self.alpha,
                       fit_intercept=self.fit_intercept,
                       copy_X=self.copy_X,
                       max_iter=self.max_iter,
                       tol=self.tol,
                       solver=self.solver,
                       positive=self.positive,
                       random_state=self.random_state)

class _ElasticNet(ElasticNet, MLRegressionConfig):
    def __init__(self, filePath):
        self.filePath = filePath
        MLRegressionConfig.__init__(self, self.filePath, "ElasticNet")
        ElasticNet.__init__(self,
                            alpha=self.alpha,
                            l1_ratio=self.l1_ratio,
                            fit_intercept=self.fit_intercept,
                            precompute=self.precompute,
                            max_iter=self.max_iter,
                            copy_X=self.copy_X,
                            tol=self.tol,
                            warm_start=self.warm_start,
                            positive=self.positive,
                            random_state=self.random_state,
                            selection=self.selection)

class _MultiTaskElasticNet(MultiTaskElasticNet, MLRegressionConfig):
    def __init__(self, filePath):
        self.filePath = filePath
        MLRegressionConfig.__init__(self, self.filePath, "MultiTaskElasticNet")
        MultiTaskElasticNet.__init__(self,
                                     alpha=self.alpha,
                                     l1_ratio=self.l1_ratio,
                                     fit_intercept=self.fit_intercept,
                                     precompute = self.precompute,
                                     max_iter=self.max_iter,
                                     tol=self.tol,
                                     warm_start=self.warm_start,
                                     positive=self.positive,
                                     random_state=self.random_state,
                                     selection=self.selection)

class _PolynomialRegression(PolynomialFeatures, MLRegressionConfig):
    def __init__(self, filePath):
        self.filePath = filePath
        MLRegressionConfig.__init__(self, self.filePath, "PolynomialRegression")
        PolynomialFeatures.__init__(self,
                                    degree=self.degree,
                                    interaction_only=self.interaction_only,
                                    include_bias=self.include_bias,
                                    order=self.order)

class _RandomForestRegression(RandomForestRegressor, MLRegressionConfig):
    def __init__(self, filePath):
        self.filePath = filePath
        MLRegressionConfig.__init__(self, self.filePath, "RandomForestRegression")
        RandomForestRegressor.__init__(self,
                                       n_estimators=self.n_estimators,
                                       criterion=self.criterion,
                                       max_depth=self.max_depth,
                                       min_samples_split=self.min_samples_split,
                                       min_samples_leaf=self.min_samples_leaf,
                                       min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                       max_features=self.max_features,
                                       max_leaf_nodes=self.max_leaf_nodes,
                                       min_impurity_decrease=self.min_impurity_decrease,
                                       bootstrap=self.bootstrap,
                                       oob_score=self.oob_score,
                                       n_jobs=self.n_jobs,
                                       random_state=self.random_state,
                                       verbose=self.verbose,
                                       warm_start=self.warm_start,
                                       ccp_alpha=self.ccp_alpha,
                                       max_samples=self.max_samples,
                                       monotonic_cst=self.monotonic_cst)

class _KNNRegression(KNeighborsRegressor, MLRegressionConfig):
    def __init__(self, filePath):
        self.filePath = filePath
        MLRegressionConfig.__init__(self, self.filePath, "KNNRegression")
        KNeighborsRegressor.__init__(self,
                                     n_neighbors=self.n_neighbors,
                                     weights=self.weights,
                                     algorithm=self.algorithm,
                                     leaf_size=self.leaf_size,
                                     p=self.p,
                                     metric=self.metric,
                                     metric_params=self.metric_params,
                                     n_jobs=self.n_jobs)

class _XGBoost(XGBRegressor, MLRegressionConfig):
    def __init__(self, filePath):
        self.filePath = filePath
        MLRegressionConfig.__init__(self, self.filePath, "XGBoost")
        XGBRegressor.__init__(self,
                              booster=self.booster,
                              verbosity=self.verbosity,
                              validate_parameters=self.validate_parameters,
                              disable_default_eval_metric=self.disable_default_eval_metric,
                              eta=self.eta,
                              gamma=self.gamma,
                              max_depth=self.max_depth,
                              min_child_weight=self.min_child_weight,
                              max_delta_step=self.max_delta_step,
                              subsample=self.subsample,
                              sampling_method=self.sampling_method,
                              colsample_bytree=self.colsample_bytree,
                              colsample_bylevel=self.colsample_bylevel,
                              colsample_bynode=self.colsample_bynode,
                              alpha=self.alpha,
                              tree_method=self.tree_method,
                              scale_pos_weight=self.scale_pos_weight,
                              refresh_leaf=self.refresh_leaf,
                              process_type=self.process_type,
                              grow_policy=self.grow_policy,
                              max_leaves=self.max_leaves,
                              max_bin=self.max_bin,
                              num_parallel_tree=self.num_parallel_tree,
                              monotone_constraints=self.monotone_constraints,
                              interaction_constraints=self.interaction_constraints,
                              multi_strategy=self.multi_strategy)