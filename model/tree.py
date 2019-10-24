import numpy as np
import lightgbm as lgb


def get_vec_name(name, size=9):
    names = []
    for i in range(size):
        for j in range(size):
            names.append(f"{name}_{i}_{j}")
    return names


COMP_BLOCK = 1000


class ModelTree:
    NUM_ROUNDS = 2 * COMP_BLOCK  # FIXME
    MODEL_PATH = "treeboost.txt"
    PARAMS = {
        "boosting_type": "dart",
        "objective": "l1",
        # "metric": "l2",
        # "objective": "binary",
        "metric": ["l1", "l2", "rmse"],
        "is_unbalance": True,
        "feature_fraction": 0.80,
        "learning_rate": 0.005,  # FIXME: 0.005 PERFECT
        "verbose": -1,
        "min_split_gain": 0.1,
        "reg_alpha": 0.2,
        "max_bin": 32,  # 512*10 FIXME/RESEARCH
        "num_leaves": 128,  # 32*10 FIXME/RESEARCH
        "max_depth": 5,
        "min_child_weight": 0.3,
        # "metric_freq": 10,
        "boost_from_average": True,
        "reg_sqrt": True,
        "zero_as_missing": True,  # FIXME:?
        "tree_learner": "data",
        "num_threads": 4,
        "snapshot_freq": COMP_BLOCK,
        # TEST
        "num_leaves": 5,
        "learning_rate": 0.05,
        "n_estimators": 720,
        "max_bin": 55,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "feature_fraction": 0.2319,
        "feature_fraction_seed": 9,
        "bagging_seed": 9,
        "min_data_in_leaf": 6,
        "min_sum_hessian_in_leaf": 11
        # "is_training_metric": "True",
    }

    def __init__(self, dataset=None):
        if dataset is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                dataset.train_and_test())

    def train(self, start, end):
        print("@train")
        lgb_train = lgb.Dataset(
            self.X_train[start:end],
            self.y_train[start:end],
            # feature_name=get_vec_name("nflh") + get_vec_name("ipar") +
            # get_vec_name("sst") + get_vec_name("pic") + get_vec_name("poc") +
            # get_vec_name("land"),
            free_raw_data=False,
        )
        print("@test")
        lgb_test = lgb.Dataset(
            self.X_test,
            self.y_test,
            # feature_name=get_vec_name("nflh") + get_vec_name("ipar") +
            # get_vec_name("sst") + get_vec_name("pic") + get_vec_name("poc") +
            # get_vec_name("land"),
            free_raw_data=False,
        )
        print("@train")
        gbm = lgb.train(
            self.PARAMS,
            lgb_train,
            init_model=self.MODEL_PATH,
            num_boost_round=self.NUM_ROUNDS,
            valid_sets=lgb_test,
            early_stopping_rounds=max(300, self.NUM_ROUNDS / 10000),
        )
        gbm.save_model(self.MODEL_PATH)
        del gbm
        # self.load()

    def load(self):
        self.pst = lgb.Booster(model_file=self.MODEL_PATH)

    def predict(self, arr):
        return self.pst.predict(arr)

    def predict_01(self, arr):
        # pred_01 = []
        return self.pst.predict(arr)
        # for x in pred:
        #    if x > 0.5:
        #        pred_01.append(False)
        #    else:
        #        pred_01.append(True)
        # return pred_01

    @staticmethod
    def get_input(x):
        return x[:, :, 1:]  # ?

    @staticmethod
    def get_output(y):
        w = np.random.uniform(0.9, 1.1)
        return np.nanmean(y[:, :, 0]) * w

    @staticmethod
    def get(cls):
        return cls.features.flatten(), cls.label

    @staticmethod
    def normalize(cls):
        cls.X = np.array(cls.X)
        cls.y = np.array(cls.y)
        # binaryzujemy wynik
        # cls.y = np.where(cls.y > cls.threshold, 0, 1)
