import os
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
from src.my_project.dataset import preprocess_for_Trainer
import torch.nn as nn
import torch
import wandb
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings
warnings.filterwarnings('ignore')
os.environ['WANDB_SILENT'] = 'true'

# 単一ラベルの他クラス分類モデル用のTrainerモデル
class SinglelabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels") # ラベルを抽出
        # クラス毎の個数を計算. (bincountは0から最大値までの個数を計算する関数)
        label_counts = torch.bincount(labels, minlength=self.model.config.num_labels)
        # 全体の個数を計算
        total_counts = torch.sum(label_counts).float()
        # 重みを計算（少数クラスほど高い重みをつける）
        class_weights = total_counts / (label_counts + 1e-6)  # 0除算を避けるために小さな数を加える
        outputs = model(**inputs)
        logits = outputs.logits
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights) 
        # loss_fct = torch.nn.CrossEntropyLoss()
        # logitsを二次元に並び替えるが，labelsは一次元．つまり，単一ラベルの他クラス分類モデルにのみ使用可能．
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 分類モデル-「活動あり」か「活動なし」を予測するモデル
class ActClassifier:
    def __init__(self, model_name, num_labels, seed):
        self.model_name = model_name
        self.num_labels = num_labels
        self.seed = seed
    # メトリクスの定義
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1) # 最大値のラベルを予測値とする
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        f1 = f1_score(y_true=labels, y_pred=predictions, average='macro')
        return {"accuracy": accuracy, "f1": f1}
    # モデル学習を行う関数
    def train_model(self, train_dataset, eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name="my-test-project", run_name='defalt'):
        # モデルの定義
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        # tokenizerの定義
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # データセットの前処理
        train_dataset = preprocess_for_Trainer(train_dataset, tokenizer, max_len=MAX_LEN)
        eval_dataset = preprocess_for_Trainer(eval_dataset, tokenizer, max_len=MAX_LEN)

        # 学習を行う時間を取得
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # wandbの初期化
        wandb.init(project=project_name, name=run_name)
        # ハイパーパラメータの設定
        training_args = TrainingArguments(
            # 学習を行った時間をファイル名にして保存
            output_dir = f"{output_dir}/{self.model_name}/{timestamp}",
            # wandbに保存
            report_to='wandb',
            # トレーニングのエポック数
            num_train_epochs = NUM_EPOCHS,
            # 訓練時のバッチサイズ
            per_device_train_batch_size = BATCH_SIZE,
            # 評価時のバッチサイズ
            per_device_eval_batch_size = BATCH_SIZE,
            # 学習率
            learning_rate = LEARNIG_RATE,
            # 乱数シード
            seed = self.seed,
            # log出力のタイミング
            logging_strategy="epoch",
            # ログの出力先
            logging_dir = f"{output_dir}/logs",
            # 進捗バーを表示するかどうか
            disable_tqdm = False,
            # 評価のタイミング
            evaluation_strategy="epoch",
            # モデルの保存タイミング
            save_strategy="epoch",
            # 常に最良のモデルを保存
            load_best_model_at_end=True,
            lr_scheduler_type="cosine",  # スケジューラタイプとして'cosine'を指定
            warmup_ratio=0.03,
            # 一つのモデルを保存
            save_total_limit = 1
        )
        # Trainerの定義
        trainer = SinglelabelTrainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = self.compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
        )
        trainer.train()
        return trainer
    # testデータを用いてラベルを予測する関数
    def predict(self, trainer, test_dataset, MAX_LEN):
        # tokenizerの定義
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # データセットの前処理
        test_dataset = preprocess_for_Trainer(test_dataset, tokenizer, max_len=MAX_LEN)
        # 予測
        predictions = trainer.predict(test_dataset)
        predictions = np.argmax(predictions.predictions, axis=-1) # 最大値のラベルを予測値とする
        return predictions
    
    # testデータを用いてモデルの評価を行う関数
    def evaluation(self, trainer, test_dataset, MAX_LEN):
        # tokenizerの定義
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # データセットの前処理
        test_dataset = preprocess_for_Trainer(test_dataset, tokenizer, max_len=MAX_LEN)
        # 評価
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
        return eval_result
    # 交差検証
    def cross_validation(self, data, test_data, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, NUM_FOLDS, output_dir, project_name):
        # 割合を固定した交差検証
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=self.seed)
        result = []
        for fold, (train_index, eval_index) in enumerate(skf.split(data['texts'], data['labels'])):
            print(f"-----------------Fold: {fold+1}-----------------")
            # 訓練データと評価データを辞書型で抽出
            train_dataset = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['labels'][i] for i in train_index]
            }
            eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['labels'][i] for i in eval_index]
            }
            # モデル学習
            trainer = self.train_model(train_dataset, eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name, run_name=f"fold_{fold+1}")
            # 評価
            eval_result = self.evaluation(trainer, test_data, MAX_LEN)
            result.append(eval_result)
            print(eval_result)
        return result
    # 交差検証
    def cross_validation_add_dataset(self, data, add_data, test_data, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, NUM_FOLDS, output_dir, project_name):
        # 割合を固定した交差検証
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=self.seed)
        result = []
        for fold, (train_index, eval_index) in enumerate(skf.split(data['texts'], data['labels'])):
            print(f"-----------------Fold: {fold+1}-----------------")
            # 訓練データと評価データを辞書型で抽出
            train_dataset = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['labels'][i] for i in train_index]
            }
            eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['labels'][i] for i in eval_index]
            }
            train_dataset['texts'] += add_data['texts']
            train_dataset['labels'] += add_data['labels']
            # モデル学習
            trainer = self.train_model(train_dataset, eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name, run_name=f"fold_{fold+1}")
            # 評価
            eval_result = self.evaluation(trainer, test_data, MAX_LEN)
            result.append(eval_result)
            print(eval_result)
        return result
    
    # 追加データを取得すると学習データに追加して学習し，そのモデルを返す関数
    def train_model_adding_data(self, train_dataset, eval_dataset, add_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name="my-test-project", run_name='defalt'):
        # train_datasetにadd_datasetを追加
        train_dataset['texts'] += add_dataset['texts']
        train_dataset['labels'] += add_dataset['labels']
        # モデルの定義
        model = self.train_model(train_dataset, eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name, run_name)
        return model
    
# 多クラスマルチラベル分類用のTrainerモデル
# 活動なしの時に細かい活動を予測するモデル
class MultilabelTrainer(Trainer):
    # 学習時の損失計算方法を定義
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels') # ラベルを抽出
        # クラス毎の個数を計算
        label_counts = labels.sum(axis=0)
        # 全体の個数を計算
        total_counts = torch.sum(label_counts)
        # 重みを計算（少数クラスほど高い重みをつける）
        class_weights = total_counts / (label_counts * len(label_counts))
        # 重みが無限大になるのを防ぐための処理
        class_weights[torch.isinf(class_weights)] = 0
        outputs = model(**inputs)
        logits = outputs.logits
        # loss_fct = nn.BCEWithLogitsLoss() # 多クラス分類の場合
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights) # 少数クラスに重みをつける場合
        # logits値と実際のラベルを用いて損失を計算
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

# 多クラスマルチ分類モデル
class MultiClassClassifier():
    def __init__(self, model_name, seed, num_labels, thresh=0.5):
        self.model_name = model_name
        self.seed = seed
        self.num_labels = num_labels
        self.thresh = thresh
    # メトリクスの定義     
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        # sigmoid関数を適応できるようにTensorに変換
        logits = torch.from_numpy(logits)
        # シグモイド関数を適用し，確率に変換
        predictions_proba = torch.sigmoid(logits)
        # 閾値を設定し予測ラベルに変換
        predictions = (predictions_proba>self.thresh).float()
        # 予測ラベルの行が0かどうかを判定するブールマスク作成
        # これを学習時に入れてしまうと，全て0になった時に学習が進まなくなる
        # mask = torch.sum(predictions, axis=1) == 0
        # # 予測ラベルが全て0のデータを除外する
        # valid_indices = ~mask
        # valid_predictions = predictions[valid_indices]
        # valid_labels = labels[valid_indices]
        # numpy.ndarray配列に変換
        # valid_predictions = valid_predictions.numpy()
        valid_labels = labels
        valid_predictions = predictions
        # accuracyを計算
        accuracy = accuracy_score(y_true=valid_labels, y_pred=valid_predictions)
        # macro f1を計算
        macro_f1 = f1_score(y_true=valid_labels, y_pred=valid_predictions, average='macro', zero_division=0)
        # クラス毎のF1値を計算
        class_f1 = [round(score, 3) for score in f1_score(y_true=valid_labels, y_pred=valid_predictions, average=None, zero_division=0)]
        # クラス毎のrecallを計算
        class_recall = [round(score, 3) for score in recall_score(y_true=valid_labels, y_pred=valid_predictions, average=None, zero_division=0)]
        # クラス毎のprecisionを計算
        class_precision = [round(score, 3) for score in precision_score(y_true=valid_labels, y_pred=valid_predictions, average=None, zero_division=0)]
        return {"accuracy": accuracy, "macro_f1": macro_f1, "class_f1": class_f1, "class_recall": class_recall, "class_precision": class_precision}
    # モデル学習を行う関数
    def train_model(self, train_dataset, eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name="my-test-project", run_name='defalt'):
        # モデルの定義
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        # tokenizerの定義
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # データセットの前処理
        train_dataset = preprocess_for_Trainer(train_dataset, tokenizer, max_len=MAX_LEN)
        eval_dataset = preprocess_for_Trainer(eval_dataset, tokenizer, max_len=MAX_LEN)

        # 学習を行う時間を取得
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # wandbの初期化
        wandb.init(project=project_name, name=run_name)
        # ハイパーパラメータの設定
        training_args = TrainingArguments(
            # 学習を行った時間をファイル名にして保存
            output_dir = f"{output_dir}/{self.model_name}/{timestamp}",
            # wandbに保存
            report_to='wandb',
            # トレーニングのエポック数
            num_train_epochs = NUM_EPOCHS,
            # 訓練時のバッチサイズ
            per_device_train_batch_size = BATCH_SIZE,
            # 評価時のバッチサイズ
            per_device_eval_batch_size = BATCH_SIZE,
            # 学習率
            learning_rate = LEARNIG_RATE,
            # 乱数シード
            seed = self.seed,
            # log出力のタイミング
            logging_strategy="epoch",
            # ログの出力先
            logging_dir = f"{output_dir}/logs",
            # 進捗バーを表示するかどうか
            disable_tqdm = False,
            # 評価のタイミング
            evaluation_strategy="epoch",
            # モデルの保存タイミング
            save_strategy="epoch",
            # 常に最良のモデルを保存
            load_best_model_at_end=True,
            lr_scheduler_type="cosine",  # スケジューラタイプとして'cosine'を指定
            # lr_scheduler_type='cosine_with_restarts', # cosタイプなので学習率は減少するが，途中でリセットされる．
            warmup_ratio=0.03,
            # 一つのモデルを保存
            save_total_limit = 1
        )
        # Trainerの定義
        trainer = MultilabelTrainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = self.compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
        )
        trainer.train()
        return trainer
    # testデータを用いてラベルを予測する関数
    def predict(self, trainer, test_dataset, MAX_LEN):
        # tokenizerの定義
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # データセットの前処理
        test_dataset = preprocess_for_Trainer(test_dataset, tokenizer, max_len=MAX_LEN)
        # 予測
        logits = trainer.predict(test_dataset).predictions
        # sigmoid関数を適応できるようにTensorに変換
        logits = torch.from_numpy(logits)
        # シグモイド関数を適用し，確率に変換
        predictions_proba = torch.sigmoid(logits)
        # 閾値を設定し予測ラベルに変換
        predictions = (predictions_proba>self.thresh).float()
        # 予測ラベルの行が0かどうかを判定するブールマスク作成
        # mask = torch.sum(predictions, dim=1) == 0
        # for i in range(len(mask)):
        #     if mask[i]:  # i番目の行が条件を満たす場合
        #         # 最大値のインデックスを取得
        #         max_index = torch.argmax(predictions_proba[i])
        #         # 最大値のインデックスの要素を1にする
        #         predictions[i][max_index] = 1
        # numpy.ndarray配列に変換
        predictions = predictions.numpy()
        return predictions
    
    # testデータを用いてモデルの評価を行う関数
    def evaluation(self, trainer, test_dataset, MAX_LEN):
        # tokenizerの定義
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # データセットの前処理
        test_dataset = preprocess_for_Trainer(test_dataset, tokenizer, max_len=MAX_LEN)
        logits = trainer.predict(test_dataset)
        # sigmoid関数を適応できるようにTensorに変換
        predictions = torch.sigmoid(torch.from_numpy(logits.predictions))
        # 閾値以上の確率を1，それ以外を0に変換
        predictions = (predictions>self.thresh).float()
        mask = torch.sum(predictions, axis=1) == 0
        # 予測ラベルが全て0のデータを除外する
        valid_indices = ~mask
        valid_predictions = predictions[valid_indices]
        labels = np.array(test_dataset['labels'])
        valid_labels = labels[valid_indices]
        # numpy.ndarray配列に変換
        valid_predictions = valid_predictions.numpy()
        # accuracyを計算
        accuracy = accuracy_score(y_true=valid_labels, y_pred=valid_predictions)
        # macro f1を計算
        macro_f1 = f1_score(y_true=valid_labels, y_pred=valid_predictions, average='macro', zero_division=0)
        # クラス毎のF1値を計算
        class_f1 = [round(score, 3) for score in f1_score(y_true=valid_labels, y_pred=valid_predictions, average=None, zero_division=0)]
        # クラス毎のrecallを計算
        class_recall = [round(score, 3) for score in recall_score(y_true=valid_labels, y_pred=valid_predictions, average=None, zero_division=0)]
        # クラス毎のprecisionを計算
        class_precision = [round(score, 3) for score in precision_score(y_true=valid_labels, y_pred=valid_predictions, average=None, zero_division=0)]
        return {"accuracy": accuracy, "macro_f1": macro_f1, "class_f1": class_f1, "class_recall": class_recall, "class_precision": class_precision}




    # 追加データを取得すると学習データに追加して学習し，そのモデルを返す関数
    def train_model_adding_data(self, train_dataset, eval_dataset, add_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name="my-test-project", run_name='defalt'):
        # train_datasetにadd_datasetを追加
        train_dataset['texts'] += add_dataset['texts']
        train_dataset['labels'] += add_dataset['labels']
        # モデルの定義
        model = self.train_model(train_dataset, eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name, run_name)
        return model

    # 交差検証
    def cross_validation(self, data, test_data, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, NUM_FOLDS, output_dir, project_name):
        # 割合を固定した交差検証
        mskf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=self.seed)
        # skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=self.seed)
        result = []
        for fold, (train_index, eval_index) in enumerate(mskf.split(data['texts'], data['labels'])):
            print(f"-----------------Fold: {fold+1}-----------------")
            # 訓練データと評価データを辞書型で抽出
            train_dataset = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['labels'][i] for i in train_index]
            }
            eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['labels'][i] for i in eval_index]
            }
            # モデル学習
            trainer = self.train_model(train_dataset, eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name, run_name=f"fold_{fold+1}")
            # 評価
            eval_result = self.evaluation(trainer, test_data, MAX_LEN)
            result.append(eval_result)
            print(eval_result)
        return result
    
    # 交差検証
    def cross_validation_add_dataset(self, data, add_data, test_data, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, NUM_FOLDS, output_dir, project_name):
        # 割合を固定した交差検証
        mskf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=self.seed)
        # skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=self.seed)
        result = []
        for fold, (train_index, eval_index) in enumerate(mskf.split(data['texts'], data['labels'])):
            print(f"-----------------Fold: {fold+1}-----------------")
            # 訓練データと評価データを辞書型で抽出
            train_dataset = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['labels'][i] for i in train_index]
            }
            eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['labels'][i] for i in eval_index]
            }
            # 追加データを学習データに追加
            train_dataset['texts'] += add_data['texts']
            train_dataset['labels'] += add_data['labels']

            # モデル学習
            trainer = self.train_model(train_dataset, eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name, run_name=f"fold_{fold+1}")
            # 評価
            eval_result = self.evaluation(trainer, test_data, MAX_LEN)
            result.append(eval_result)
            print(eval_result)
        return result
    

# 多クラスマルチラベル分類用のTrainerモデルv2-活動なし予測後に細かい活動を予測するに段階予測
# 問題点：二値分類-マルチラベル分類を行う時に二つの分類器あり．
# 下のコードでは，wandbの影響から，予測を行う際にモデルを二つ保存できない．モデルを二つ保存し，それを用いて予測を行えるように変更する．
class ActivityPrediction():
    def __init__(self, model_name, seed, num_labels, thresh=0.5):
        self.model_name = model_name # モデル名
        self.seed = seed # 乱数シード
        self.num_labels = num_labels # ラベル数
        self.thresh = thresh # 閾値
    
    # test_datasetの中身はmulti用のラベルを含む辞書型のデータ
    def train_and_predict(self, single_train_dataset, single_eval_dataset, multi_train_dataset, multi_eval_dataset, single_test_dataset, multi_test_dataset, MAX_LEN, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name):
        # ラベルの答え
        answer = multi_test_dataset['texts']
        # answerリストと同じ二次元のリストを作成.ただし，全ての要素を0にする
        prediction_label = [[0 for i in range(len(answer[0]))] for j in range(len(answer))]

        # ステップ1: '活動あり/なし'のモデル作成
        # '活動あり/なし'を予測するためのインスタンス作成
        single_label_classifier = ActClassifier(self.model_name, 2, self.seed)
        # single用のモデル作成
        model1 = single_label_classifier.train_model(single_train_dataset, single_eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name)
        # ステップ２: 細かい行動のモデル作成
        # '活動あり'のデータのみで細かい活動を予測するためのインスタンス作成
        multi_label_classifier = MultiClassClassifier(self.model_name, self.seed, self.num_labels-1, self.thresh)
        # multi用のデータから'活動なし'のデータを除外.かつ，最終列を削除
        multi_train_dataset = self.filter_and_remove_last_column(multi_train_dataset)
        multi_eval_dataset = self.filter_and_remove_last_column(multi_eval_dataset)
        # multi用のモデル作成
        model2 = multi_label_classifier.train_model(multi_train_dataset, multi_eval_dataset, MAX_LEN, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name)
        # 予測
        # model1: 2値分類器, model2: 行動なし以外の行動を予測する分類器
        prediction_label = self.predict(model1, model2, multi_test_dataset, MAX_LEN)    
        return prediction_label
    

    # ニクラス分類モデルと，他クラスマルチラベル分類モデルを引数として受け取った時に予測ラベルを返す関数
    # モデルはどちらもTrainer型
    # test_datasetは'text'カラムを含むDataFrame型のデータ
    def predict(self, model1, model2, test_dataset, MAX_LEN, thresh=0.5):
        # 2つのモデルとテストデータ，テストデータを数字に変換するために必要なMAX_LEN，閾値を引数として受け取る
        # 使用するtokenizerを定義
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # test_datasetからtextsカラムのみの辞書型データに変換
        test_dataset = {'texts': test_dataset['texts']}

        # test_datasetをTrainerで使用できる形に変換
        test_dataset = preprocess_for_Trainer(test_dataset, tokenizer, max_len=MAX_LEN)

        # 予測ラベルを保存するリストを作成．ただし，全ての要素を0にする
        # テストサイズ×マルチラベル数の二次元リスト
        prediction_label = [[0 for i in range(self.num_labels)] for j in range(len(test_dataset))]

        # step1: '活動あり/なし'の予測
        # 活動あり/なしの予測確率
        pred_proba_2class = model1.predict(test_dataset)
        # 2classの大きい方の確率を予測値とする
        prediction_2class = np.argmax(pred_proba_2class.predictions, axis=-1) # 最大値のラベルを予測値とする

        # prediction_2classが0のprediction_labelのindexの最終列に1を追加
        for i in range(len(prediction_2class)):
            if prediction_2class[i] == 0:
                prediction_label[i][-1] = 1
                        
        # step2: '活動あり'のデータのみで細かい活動を予測
        # '活動あり'のデータのみを抽出
        new_test_dataset = self.extract_prediction_activity_data_v2(test_dataset, prediction_2class)
        # new_test_datasetをTrainerで使用できる形に変換
        new_test_dataset = preprocess_for_Trainer(new_test_dataset, tokenizer, max_len=MAX_LEN)
        # 細かい活動の予測値を算出
        pred_proba_multi = model2.predict(new_test_dataset)
        # 予測値にsigmoid関数を適応し，確率に変換
        pred_proba_multiclass = torch.sigmoid(torch.from_numpy(pred_proba_multi.predictions))
        # 閾値以上の確率を1，それ以外を0に変換
        prediction_multiclass = (pred_proba_multiclass>self.thresh).int().numpy()

        # 予測結果をリストに追加
        j = 0
        for i in range(len(prediction_2class)):
            if prediction_2class[i] == 1: # 活動ありと予測された行にのみ予測値を追加
                # prediction_2classが1のprediction_labelのindexの最終列までにprediction_multiclassの値を追加
                prediction_label[i][:-1] = prediction_multiclass[j]
                j += 1
        return prediction_label

    # 交差検証
    def cross_validation(self, single_data, multi_data, single_test_data, multi_test_data,MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, NUM_FOLDS, output_dir, project_name):
        # 割合を固定した交差検証
        mskf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=self.seed)
        # skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=self.seed)
        result = []
        for fold, (train_index, eval_index) in enumerate(mskf.split(multi_data['texts'], multi_data['labels'])):
            print(f"-----------------Fold: {fold+1}-----------------")
            # 訓練データと評価データを辞書型で抽出
            single_train_dataset = {
                'texts': [single_data['texts'][i] for i in train_index],
                'labels': [single_data['labels'][i] for i in train_index]
            }
            multi_train_dataset = {
                'texts': [multi_data['texts'][i] for i in train_index],
                'labels': [multi_data['labels'][i] for i in train_index]
            }
            single_eval_dataset = {
                'texts': [single_data['texts'][i] for i in eval_index],
                'labels': [single_data['labels'][i] for i in eval_index]
            }
            multi_eval_dataset = {
                'texts': [multi_data['texts'][i] for i in eval_index],
                'labels': [multi_data['labels'][i] for i in eval_index]
            }

            # 2stepモデル学習
            prediction_label = self.train_and_predict(single_train_dataset, single_eval_dataset, multi_train_dataset, multi_eval_dataset, single_test_data, multi_test_data, MAX_LEN, NUM_EPOCHS, LEARNIG_RATE, BATCH_SIZE, PATIENCE, output_dir, project_name)
            # 評価
            eval_result = self.evaluation(multi_test_data['labels'], prediction_label)
            # 評価結果をリストに追加
            result.append(eval_result)
            print(eval_result)
        return result

    def evaluation(self, true_label, prediction_label):
        # prediction_labelの行が全て0の場合はその行以外で評価を行う
        mask = np.sum(prediction_label, axis=1) == 0
        true_label = np.array(true_label)
        prediction_label = np.array(prediction_label)
        true_label = true_label[~mask]
        prediction_label = prediction_label[~mask]
        # accuracyを計算
        accuracy = accuracy_score(y_true=true_label, y_pred=prediction_label)
        # macro f1を計算
        macro_f1 = f1_score(y_true=true_label, y_pred=prediction_label, average='macro', zero_division=0)
        # クラス毎のF1値を計算
        class_f1 = f1_score(y_true=true_label, y_pred=prediction_label, average=None, zero_division=0)
        # クラス毎のF1値を計算
        class_f1 = [round(score, 3) for score in f1_score(y_true=true_label, y_pred=prediction_label, average=None, zero_division=0)]
        # クラス毎のrecallを計算
        class_recall = [round(score, 3) for score in recall_score(y_true=true_label, y_pred=prediction_label, average=None, zero_division=0)]
        # クラス毎のprecisionを計算
        class_precision = [round(score, 3) for score in precision_score(y_true=true_label, y_pred=prediction_label, average=None, zero_division=0)]

        return {"accuracy": accuracy, "macro_f1": macro_f1, "class_f1": class_f1, 'class_recall': class_recall, 'class_precision': class_precision}
        # return {"accuracy": accuracy, "f1": f1}
    
    def extract_prediction_activity_data(self, test_dataset, single_label_result):
        # '活動あり'と予測されたデータを抽出する関数
        # 'single_label_result'が1の部分のtest_datasetを抽出する関数
        # test-datasetは['texts','labels']を含む辞書型のデータ
        new_test_dataset = {
            'texts': [],  # 初期化
            'labels': []  # 初期化
        }
        
        for text, label, prediction in zip(test_dataset['texts'], test_dataset['labels'], single_label_result):
            if prediction == 1:  # 1は'活動あり'を示す
                new_test_dataset['texts'].append(text)  # テキストを追加
                new_test_dataset['labels'].append(label[:-1])  # 最終列以外のラベルを追加

        return new_test_dataset
    def extract_prediction_activity_data_v2(self, test_dataset, single_prediction_label):
        # '活動あり'と予測されたデータを抽出する関数
        # 'single_label_result'が1の部分のtest_datasetを抽出する関数
        # test-datasetは['texts']を含む辞書型のデータ
        new_test_dataset = {
            'texts': []  # 初期化
        }
        
        for text, prediction in zip(test_dataset['texts'], single_prediction_label):
            if prediction == 1: # 1は'活動あり'を示す
                new_test_dataset['texts'].append(text)  # テキストを追加
        
        return new_test_dataset

    # 活動なしのデータを含めたデータを活動ありのみのデータに変換する関数
    def filter_and_remove_last_column(self, data):
        # 最終列が1でない行のみを保持
        filtered_data = [(text, label) for text, label in zip(data['texts'], data['labels']) if label[-1] != 1]

        # 'texts'と'labels'を更新し、最終列を削除
        updated_texts, updated_labels = zip(*[(text, label[:-1]) for text, label in filtered_data])

        # 新しい辞書を作成して返す
        return {
            'texts': list(updated_texts),
            'labels': list(updated_labels)
        }

        
    
