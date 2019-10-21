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
        "boosting_type": "gbdt",
        "objective": "l1",
        # "metric": "l2",
        # "objective": "binary",
        "metric": ["l1", "rmse"],
        "is_unbalance": True,
        "feature_fraction": 0.80,
        "learning_rate": 0.005,
        "verbose": -1,
        "min_split_gain": 0.05,
        "reg_alpha": 0.2,
        "max_bin": 256,  # 512*10 FIXME/RESEARCH
        "num_leaves": 32,  # 32*10 FIXME/RESEARCH
        "max_depth": 5,
        "min_child_weight": 0.3,
        # "metric_freq": 10,
        "boost_from_average": True,
        "reg_sqrt": True,
        "zero_as_missing": True,  # FIXME:?
        "tree_learner": "feature",
        "num_threads": 4,
        "snapshot_freq": COMP_BLOCK,
        # "is_training_metric": "True",
    }

    def __init__(self, dataset=None):
        if dataset is not None:
            X_train, X_test, y_train, y_test = dataset.train_and_test()
            self.lgb_train = lgb.Dataset(
                X_train,
                y_train,
                feature_name=get_vec_name("nflh") + get_vec_name("ipar") +
                get_vec_name("sst") + get_vec_name("pic") +
                get_vec_name("poc") + get_vec_name("land"),
            )
            self.lgb_test = lgb.Dataset(
                X_test,
                y_test,
                feature_name=get_vec_name("nflh") + get_vec_name("ipar") +
                get_vec_name("sst") + get_vec_name("pic") +
                get_vec_name("poc") + get_vec_name("land"),
            )

    def train(self):
        gbm = lgb.train(
            self.PARAMS,
            self.lgb_train,
            init_model=self.MODEL_PATH,
            num_boost_round=self.NUM_ROUNDS,
            valid_sets=self.lgb_test,
            early_stopping_rounds=max(300, self.NUM_ROUNDS / 10000),
        )
        gbm.save_model(self.MODEL_PATH)
        self.load()

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
        return x[:, :, 1:]

    @staticmethod
    def get_output(y):
        return np.nanmean(y[:, :, 0])

    @staticmethod
    def get(cls):
        return cls.features.flatten(), cls.label

    @staticmethod
    def normalize(cls):
        cls.X = np.array(cls.X)
        cls.y = np.array(cls.y)
        # binaryzujemy wynik
        # cls.y = np.where(cls.y > cls.threshold, 0, 1)
