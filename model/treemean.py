import numpy as np
import lightgbm as lgb


class ModelTreeMean:
    NUM_ROUNDS = 100 * 50000
    MODEL_PATH = "treemeanboost.txt"
    PARAMS = {
        "boosting_type": "gbdt",
        # "objective": "regression",
        # "metric": "l2",
        "objective": "binary",
        "metric": ["binary_error", "binary_logloss"],
        "is_unbalance": True,
        "feature_fraction": 0.85,
        "learning_rate": 0.005,
        "verbose": -1,
        "min_split_gain": 0.1,
        "reg_alpha": 0.3,
        "max_bin": 512,  # 512*10 FIXME/RESEARCH
        "num_leaves": 32,  # 32*10 FIXME/RESEARCH
        "max_depth": 9,
        "min_child_weight": 0.5,
        "is_training_metric": "True",
    }

    def __init__(self, dataset=None):
        if dataset is not None:
            X_train, X_test, y_train, y_test = dataset.train_and_test()
            print(f"---------------> \033[92m {X_train.shape}\033[m")
            self.lgb_train = lgb.Dataset(
                X_train,
                y_train,
                feature_name=[
                    "chlor_a",
                    "nflh",
                    "ipar",
                    "sst",
                    "pic",
                    "poc",
                    "land",
                ],
            )
            self.lgb_test = lgb.Dataset(
                X_test,
                y_test,
                feature_name=[
                    "chlor_a",
                    "nflh",
                    "ipar",
                    "sst",
                    "pic",
                    "poc",
                    "land",
                ],
            )

    def train(self):
        gbm = lgb.train(
            self.PARAMS,
            self.lgb_train,
            num_boost_round=self.NUM_ROUNDS,
            valid_sets=self.lgb_test,
            early_stopping_rounds=self.NUM_ROUNDS / 10000,
        )
        gbm.save_model(self.MODEL_PATH)
        self.load()

    def load(self):
        self.pst = lgb.Booster(model_file=self.MODEL_PATH)

    def predict(self, arr):
        return self.pst.predict(arr)

    def predict_01(self, arr):
        pred_01 = []
        pred = self.pst.predict(arr)
        for x in pred:
            if x > 0.5:
                pred_01.append(True)
            else:
                pred_01.append(False)
        return pred_01

    @staticmethod
    def get_input(x):  # (0, 1) correct, (2) better
        return np.mean(x[:, :, 0:], axis=(0, 1))

    @staticmethod
    def get_output(y):
        return np.nanmean(y[:, :, 0])

    def get(self, cls):
        return cls.features.flatten(), cls.label

    @staticmethod
    def normalize(cls):
        cls.X = np.array(cls.X)
        cls.y = np.array(cls.y)
        # binaryzujemy wynik
        cls.y = np.where(cls.y > cls.threshold, 0, 1)
