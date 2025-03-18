import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from transformers import BertTokenizer, BertForSequenceClassification
from src.my_project.dataset import preprocess_dataset, preprocess_multiclass_dataset, preprocess_text_dataset
from src.my_project.utils import set_seed
from skmultilearn.model_selection import IterativeStratification
import numpy as np

# 分類モデル
class ActClassifier:
    # 分類モデルの訓練
    def __init__(self,model_name, tokenizer, criterion, device, seed=2023):
        set_seed(seed)
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.device = device
        self.optimizer = None

    def train_model(self, train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE, patience=3):
        model.train()
        # AdamWオプティマイザを使用してモデルのパラメータを最適化．パラメータを徐々に変えていくもの．
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # model.parameters()でモデルの全てのパラメータを取得
        # 無限大（infinity）に初期化
        best_loss = float('inf')
        current_patience = 0

        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # ミニバッチ数×105のテンソルの二次元配列
                input_ids = batch[0].to(device)
                # ミニバッチ数×105のテンソルの二次元配列．attention_mask
                attention_mask = batch[1].to(device)
                # ミニバッチ数の正解ラベルのテンソル型の一次元配列
                labels = batch[2].to(device) 
                # 勾配をゼロにリセット
                self.optimizer.zero_grad()
                # BERTへの入力
                outputs = model(input_ids, attention_mask=attention_mask)
                # ミニバッチサイズ×ラベル数のテンソル型の二次元配列．各文章がどのラベルの確率が高いか
                logits = outputs.logits
                # 実際のラベルとの誤差算出
                loss = self.criterion(logits, labels)
                total_loss += loss.item() # 誤差加算
                # 損失関数の勾配を計算
                loss.backward()
                # パラメータの最適化実行
                self.optimizer.step()
            # len(train_dataloder)はバッチの数．1epoch毎のloss
            avg_loss = total_loss / len(train_dataloader)
            # print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # 早期終了のチェック
            if avg_loss < best_loss:
                best_loss = avg_loss
                current_patience = 0
            else:
                # 1エポックのlossがbest lossより大きい時current_patienceに1を加算
                current_patience += 1
                # 連続でlossが増加して，patience回数を超えた時に学習を終了させる
                if current_patience >= patience:
                    print("Early stopping. No improvement in loss.")
                    break
        return epoch + 1

    # モデルの評価
    def evaluate_model(self, eval_dataloader, model, best_f1, best_model, device):
        model.eval()
        total_predictions = []
        total_labels = []

        # 推論時はtorch.no_grad()の中でする
        with torch.no_grad():
            for input_ids, attention_mask, labels in eval_dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask)
                _, predictions = torch.max(outputs.logits, dim=1)

                total_predictions.extend(predictions.detach().cpu().numpy())
                total_labels.extend(labels.detach().cpu().numpy())

        accuracy = accuracy_score(total_labels, total_predictions)
        f1 = f1_score(total_labels, total_predictions, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

        return accuracy, f1, best_f1, best_model, total_predictions, total_labels

    # 交差検証
    def cross_validation(self, model_name, data, num_folds, NUM_EPOCHS, device,
                        LEARNING_RATE, BATCH_SIZE, output_dir, MAX_LEN, timestamp, num_labels):
        # 割合を固定したまま，ランダムに被らないようにデータを分ける．
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2023) # random_stateを決めてため結果は変わらない

        fold_scores = []
        fold_f1 = []
        best_model = None
        best_f1 = 0.0
        finished_epochs = []
        all_predictions = []
        all_labels = []

        for fold, (train_index, eval_index) in enumerate(skf.split(data['texts'], data['labels'])):
            print(f"-----------------Fold: {fold+1}-----------------")

            # 文章分類を行うクラス：BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            # モデルをGPUに配置
            model.to(device)
            # 訓練データと評価データを辞書型で抽出
            train_dataset = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['labels'][i] for i in train_index]
            }
            eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['labels'][i] for i in eval_index]
            }

            # ここでデータセットの前処理を行っている．
            train_encodings = preprocess_dataset(train_dataset, self.tokenizer, MAX_LEN)
            eval_encodings = preprocess_dataset(eval_dataset, self.tokenizer, MAX_LEN)
            # データローダ型に変形．ミニバッチごとのデータに分割
            train_dataloader = DataLoader(train_encodings, BATCH_SIZE, shuffle=True)
            eval_dataloader = DataLoader(eval_encodings, BATCH_SIZE)

            # 学習させ学習が終わるエポックを返す
            finished_epoch = self.train_model(train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE)
            finished_epochs.append(finished_epoch)
            # 評価
            accuracy, f1, best_f1, best_model, predictions, labels = self.evaluate_model(eval_dataloader, model, best_f1, best_model, device)
            fold_scores.append(accuracy)
            fold_f1.append(f1)
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            print(f"Accuracy: {accuracy}")
            print(f"macro f1: {f1}")

        average_accuracy = sum(fold_scores) / num_folds
        average_f1 = sum(fold_f1) / num_folds

        print("-------------------------------------")
        print(f"finished epochs : {finished_epochs}")
        print(f"Average Accuracy: {average_accuracy}")
        print(f"Average macro f1: {average_f1}")

        # モデルの保存
        best_model.save_pretrained(f"{output_dir}/{model_name}{timestamp}")

        return fold_scores, fold_f1, best_model, all_predictions, all_labels  # 予測値と実際のラベルを返す
    
    # 全データ学習
    def train_model_all_data(self, model_name, data, NUM_EPOCHS, device,
                        LEARNING_RATE, BATCH_SIZE, output_dir, MAX_LEN, timestamp, num_labels):

        # 文章分類を行うクラス：BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        # モデルをGPUに配置
        model.to(device)
        # 訓練データを辞書型で抽出
        train_dataset = {
            'texts': data['texts'],
            'labels': data['labels']
        }
        self.MAX_LEN = MAX_LEN
        self.BATCH_SIZE = BATCH_SIZE
        # ここでデータセットの前処理を行っている．
        train_encodings = preprocess_dataset(train_dataset, self.tokenizer, MAX_LEN)
        # データローダ型に変形．ミニバッチごとのデータに分割
        train_dataloader = DataLoader(train_encodings, BATCH_SIZE, shuffle=True)

        # 学習させ学習が終わるエポックを返す
        finished_epoch = self.train_model(train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE)

        print("-------------------------------------")
        print(f"finished epochs : {finished_epoch}")

        # モデルの保存
        model.save_pretrained(f"{output_dir}/{model_name}{timestamp}")

        # モデルをクラス内に保存
        self.model = model
        return model
    
    # testデータに対して確率,クラスを返す
    def prediction(self, test_data):
        self.model.eval()
        # 予測ラベルを格納するリスト
        total_labels = []
        # 確率を格納するリスト
        total_probabilities = []
        # 訓練データを辞書型で抽出
        test_dataset = {
            'texts': test_data['texts']
        }
        # ここでデータセットの前処理を行っている．
        test_encodings = preprocess_text_dataset(test_dataset, self.tokenizer, self.MAX_LEN)
        # データローダ型に変形．ミニバッチごとのデータに分割
        # shuffleをFalseにすることで，testデータの順番を保持する
        test_dataloader = DataLoader(test_encodings, self.BATCH_SIZE, shuffle=False)
        # 推論時はtorch.no_grad()の中でする
        with torch.no_grad():
            for input_ids, attention_mask in test_dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                # BERTへ入力
                outputs = self.model(input_ids, attention_mask)
                # 確率を返す
                probabilities = F.softmax(outputs.logits, dim=1)
                # 確率とそのindexを返す
                _, predictions = torch.max(probabilities, dim=1)
                #　予測ラベルをリストに追加
                total_labels.extend(predictions.detach().cpu().numpy())  
                total_probabilities.extend(probabilities.detach().cpu().numpy())
        # numpy配列に変換
        total_labels = np.array(total_labels)
        total_probabilities = np.array(total_probabilities)
        # 各ラベルになる確率と，最大ラベルを返す
        return total_probabilities,total_labels
    
    # np.ndarray配列の中で確率が閾値以上のテキストとそのクラスを辞書型で返す
    def get_text_and_label(self, test_data, total_probabilities, total_labels, threshold):
        # 閾値以上の確率を持つindexとラベルを返す
        index,label = np.where(total_probabilities >= threshold)
        # 閾値以上の確率を持つデータのtextを抽出
        selected_texts = [test_data['texts'][i] for i in index]
        # 閾値以上の確率を持つデータのラベルを抽出
        selected_labels = total_labels[index].tolist()
        # 閾値以上の確率を持つデータのtextとlabelを辞書型で返す
        text_and_label = {
            'texts': selected_texts,
            'labels': selected_labels
        }
        return text_and_label

    # 半教師あり学習を行いその評価を行う
    # 交差検証
    def Semi_Supervised_Learning_cross_validation(self, model_name, data, add_data, num_folds, NUM_EPOCHS, device,
                        LEARNING_RATE, BATCH_SIZE, output_dir, MAX_LEN, timestamp, num_labels):
        # 割合を固定したまま，ランダムに被らないようにデータを分ける．
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2023)

        fold_scores = []
        fold_f1 = []
        best_model = None
        best_f1 = 0.0
        finished_epochs = []
        all_predictions = []
        all_labels = []

        for fold, (train_index, eval_index) in enumerate(skf.split(data['texts'], data['labels'])):
            print(f"-----------------Fold: {fold+1}-----------------")

            # 文章分類を行うクラス：BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            # モデルをGPUに配置
            model.to(device)
            # 訓練データと評価データを辞書型で抽出
            trained_dataset = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['labels'][i] for i in train_index]
            }
            eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['labels'][i] for i in eval_index]
            }
            train_dataset = {
                'texts': trained_dataset['texts'] + add_data['texts'],
                'labels': trained_dataset['labels'] + add_data['labels']
            }

            # ここでデータセットの前処理を行っている．
            train_encodings = preprocess_dataset(train_dataset, self.tokenizer, MAX_LEN)
            eval_encodings = preprocess_dataset(eval_dataset, self.tokenizer, MAX_LEN)
            # データローダ型に変形．ミニバッチごとのデータに分割
            train_dataloader = DataLoader(train_encodings, BATCH_SIZE, shuffle=True)
            eval_dataloader = DataLoader(eval_encodings, BATCH_SIZE)

            # 学習させ学習が終わるエポックを返す
            finished_epoch = self.train_model(train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE)
            finished_epochs.append(finished_epoch)
            # 評価
            accuracy, f1, best_f1, best_model, predictions, labels = self.evaluate_model(eval_dataloader, model, best_f1, best_model, device)
            fold_scores.append(accuracy)
            fold_f1.append(f1)
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            print(f"Accuracy: {accuracy}")
            print(f"macro f1: {f1}")

        average_accuracy = sum(fold_scores) / num_folds
        average_f1 = sum(fold_f1) / num_folds

        print("-------------------------------------")
        print(f"finished epochs : {finished_epochs}")
        print(f"Average Accuracy: {average_accuracy}")
        print(f"Average macro f1: {average_f1}")

        # モデルの保存
        best_model.save_pretrained(f"{output_dir}/{model_name}{timestamp}")

        return fold_scores, fold_f1, best_model, all_predictions, all_labels  # 予測値と実際のラベルを返す
    
class MultiClassActClassifier:
# マルチ分類モデルの訓練
    def __init__(self, model_name, tokenizer, device, seed=2023):
        set_seed(seed)
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.criterion = torch.nn.BCEWithLogitsLoss()  # 損失関数を定義
        self.device = device
        self.optimizer = None

    def train_model(self, train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE, patience=3):
            model.train()
            # AdamWオプティマイザを使用してモデルのパラメータを最適化するための設定．パラメータを徐々に変えていくもの．
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # model.parameters()でモデルの全てのパラメータを取得
            best_loss = float('inf') # 無限大（infinity）に初期化
            current_patience = 0

            for epoch in range(NUM_EPOCHS):
                total_loss = 0
                for step, batch in enumerate(train_dataloader):
                    input_ids = batch[0].to(device) # ミニバッチサイズの個数×105のテンソルの二次元配列．これがBERTへの入力
                    attention_mask = batch[1].to(device) # ミニバッチサイズの個数×105のテンソルの二次元配列．attention_mask.
                    labels = batch[2].to(device) # ミニバッチサイズの個数の正解ラベルでテンソル型の一次元配列

                    # 勾配をゼロにリセット
                    self.optimizer.zero_grad()
                    # BERTへ入力
                    outputs = model(input_ids, attention_mask=attention_mask)
                    # ミニバッチサイズ×ラベル数のテンソル型の二次元配列．各文章毎の各クラススコアが算出(確率ではない)
                    logits = outputs.logits
                    # 実際の値とスコアとの差を算出．実際は内部的にsoftmaxをかけているため，確率として扱うことができる
                    loss = self.criterion(logits, labels.float())
                    # 誤差加算
                    total_loss += loss.item()
                    # 損失に基づいて勾配を計算
                    loss.backward()
                    # 勾配を使用しパラメータを更新
                    self.optimizer.step()

                avg_loss = total_loss / len(train_dataloader) # len(train_dataloder)はバッチの数
                # print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

                # 早期終了のチェック
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    current_patience = 0
                else:
                    current_patience += 1
                    if current_patience >= patience:
                        print("Early stopping. No improvement in loss.")
                        break

            return epoch + 1

    # モデルの評価
    def evaluate_model(self, eval_dataloader, model, best_f1, best_model, device):
        model.eval()
        total_predictions = []
        total_labels = []

        # 予測
        with torch.no_grad():
            for input_ids, attention_mask, labels in eval_dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask)
                predictions = torch.sigmoid(outputs.logits)  # シグモイド活性化関数の適用
                # predictions = outputs.logits
                predictions = (predictions > 0.5).float()  # 0.5を閾値として二値化

                total_predictions.extend(predictions.detach().cpu().numpy())
                total_labels.extend(labels.detach().cpu().numpy())

        accuracy = accuracy_score(total_labels, total_predictions)
        f1 = f1_score(total_labels, total_predictions, average='macro', zero_division=0)  # マクロ平均F1スコアの計算
        hamming = hamming_loss(total_labels, total_predictions)  # ハムミング損失の計算

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

        return accuracy, f1, hamming, best_f1, best_model, total_predictions, total_labels  # ハムミング損失を返す

    # 交差検証
    def cross_validation(self, model_name, data, num_folds, NUM_EPOCHS, device,
                        LEARNING_RATE, BATCH_SIZE, output_dir, MAX_LEN, timestamp, num_labels):
        # 割合を固定したまま，ランダムに被らないようにデータを分ける．
        skf = KFold(n_splits=num_folds, shuffle=True, random_state=2023)
        # skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2023)
        # skf = IterativeStratification(n_splits=num_folds, shuffle=True, random_state=2023)

        fold_scores = []
        fold_f1 = []
        fold_hamming = []
        best_model = None
        best_f1 = 0.0
        finished_epochs = []
        all_predictions = []
        all_labels = []

        for fold, (train_index, eval_index) in enumerate(skf.split(data['texts'], data['labels'])):
            print(f"-----------------Fold: {fold+1}-----------------")

            # 文章分類を行うクラス：BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            # モデルをGPUに配置
            model.to(device)
            # 訓練データと評価データを辞書型で抽出
            train_dataset = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['labels'][i] for i in train_index]
            }
            eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['labels'][i] for i in eval_index]
            }

            # ここでデータセットの前処理を行っている．
            train_encodings = preprocess_multiclass_dataset(train_dataset, self.tokenizer, MAX_LEN)
            eval_encodings = preprocess_multiclass_dataset(eval_dataset, self.tokenizer, MAX_LEN)
            # データローダ型に変形．ミニバッチごとのデータに分割
            train_dataloader = DataLoader(train_encodings, BATCH_SIZE, shuffle=True)
            eval_dataloader = DataLoader(eval_encodings, BATCH_SIZE)

            # 学習させ学習が終わるエポックを返す
            finished_epoch = self.train_model(train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE)
            finished_epochs.append(finished_epoch)
            # 評価
            accuracy, f1, hamming, best_f1, best_model, predictions, labels = self.evaluate_model(eval_dataloader, model, best_f1, best_model, device)
            fold_scores.append(accuracy)
            fold_f1.append(f1)
            fold_hamming.append(hamming)
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            print(f"Accuracy: {accuracy}")
            print(f"macro f1: {f1}")
            print(f"humming: {hamming}")

        average_accuracy = sum(fold_scores) / num_folds
        average_f1 = sum(fold_f1) / num_folds
        average_hamming = sum(fold_hamming) /num_folds

        print("-------------------------------------")
        print(f"finished epochs : {finished_epochs}")
        print(f"Average Accuracy: {average_accuracy}")
        print(f"Average macro f1: {average_f1}")
        print(f'Average humming: {average_hamming}')

        # モデルの保存
        best_model.save_pretrained(f"{output_dir}/{model_name}{timestamp}")

        return fold_scores, fold_f1, fold_hamming, best_model, all_predictions, all_labels  # 予測値と実際のラベルを返す
    
    # 全データを使用しモデルを学習させる
    def train_full_dataset(self, model_name, data, NUM_EPOCHS, device, LEARNING_RATE, BATCH_SIZE, output_dir, MAX_LEN, timestamp, num_labels):
        print(f"----------------- Full Training -----------------")

        # 文章分類を行うクラス：BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        # モデルをGPUに配置
        model.to(device)

        # ここでデータセットの前処理を行っている．
        encodings = preprocess_multiclass_dataset(data, self.tokenizer, MAX_LEN)
        # データローダ型に変形．ミニバッチごとのデータに分割
        dataloader = DataLoader(encodings, BATCH_SIZE, shuffle=True)

        # 学習させ学習が終わるエポックを返す
        self.train_model(dataloader, model, NUM_EPOCHS, device, LEARNING_RATE)

        # モデルの保存
        model.save_pretrained(f"{output_dir}/{model_name}{timestamp}")

        print(f"Training complete. Model saved to {output_dir}/{model_name}{timestamp}")

        return model  # 学習済みモデルを返す

# モデルとデータを使用し，予測値を返す関数
class ModelPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    # 与えられたデータをエンコーディングし，その後データローダ型に変換し，モデルに入力し予測値を返す
    def predict(self, data, tokenizer, MAX_LEN, BATCH_SIZE):
        # ここでデータセットの前処理を行っている．
        encodings = preprocess_text_dataset(data, tokenizer, MAX_LEN)
        # データローダ型に変形．ミニバッチごとのデータに分割
        dataloader = DataLoader(encodings, BATCH_SIZE)

        # モデルをGPUに配置
        self.model.to(self.device)
        self.model.eval()  # モデルを評価モードに設定
        total_predictions = []

        # 予測の実行
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(dataloader):  # ラベルは使用しないため、'_' とします
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids, attention_mask)
                predictions = outputs.logits
                predictions = (predictions > 0.5).float()  # 0.5を閾値として二値化

                total_predictions.extend(predictions.detach().cpu().numpy())

        return total_predictions  # 予測されたラベルのリストを返す



# 学習1: ActClassifierで訓練データを活動あり，活動なしの二値学習(model1)
# 学習2: 訓練データの活動ありのデータに20種類に分類する学習(model2)
# 評価フロー: テストデータをmodel1を用い活動ありかの予測，テストデータにmodel2を用いてマルチ分類するようなクラスを作成
class ActMulticlassifier:
    # マルチ分類モデルの訓練
    def __init__(self, model_name, tokenizer, device, seed=2023):
        set_seed(seed)
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.criterion = torch.nn.BCEWithLogitsLoss()  # 損失関数を定義
        self.device = device
        self.optimizer = None

    def train_2_model(self, train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE, patience=3):
        model.train()
        # AdamWオプティマイザを使用してモデルのパラメータを最適化するための設定．パラメータを徐々に変えていくもの．
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # model.parameters()でモデルの全てのパラメータを取得
        best_loss = float('inf') # 無限大（infinity）に初期化
        current_patience = 0

        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                input_ids = batch[0].to(device) # ミニバッチサイズの個数×105のテンソルの二次元配列．これがBERTへの入力
                attention_mask = batch[1].to(device) # ミニバッチサイズの個数×105のテンソルの二次元配列．attention_mask.
                labels = batch[2].to(device) # ミニバッチサイズの個数の正解ラベルでテンソル型の一次元配列

                # 勾配をゼロにリセット
                self.optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask) # BERTへ入力
                logits = outputs.logits # ミニバッチサイズ×ラベル数のテンソル型の二次元配列．各文章がどのラベルの確率が高いか
                loss = self.criterion(logits, labels) # 実際の値との誤差

                total_loss += loss.item() # 誤差加算
                # 損失関数の勾配を計算
                loss.backward()
                # パラメータの最適化実行
                self.optimizer.step()

            avg_loss = total_loss / len(train_dataloader) # len(train_dataloder)はバッチの数
            # print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # 早期終了のチェック
            if avg_loss < best_loss:
                best_loss = avg_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print("Early stopping. No improvement in loss.")
                    break

        return epoch + 1

    def train_multi_model(self, train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE, patience=3):
            model.train()
            # AdamWオプティマイザを使用してモデルのパラメータを最適化するための設定．パラメータを徐々に変えていくもの．
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # model.parameters()でモデルの全てのパラメータを取得
            best_loss = float('inf') # 無限大（infinity）に初期化
            current_patience = 0

            for epoch in range(NUM_EPOCHS):
                total_loss = 0
                for step, batch in enumerate(train_dataloader):
                    input_ids = batch[0].to(device) # ミニバッチサイズの個数×105のテンソルの二次元配列．これがBERTへの入力
                    attention_mask = batch[1].to(device) # ミニバッチサイズの個数×105のテンソルの二次元配列．attention_mask.
                    labels = batch[2].to(device) # ミニバッチサイズの個数の正解ラベルでテンソル型の一次元配列

                    # 勾配をゼロにリセット
                    self.optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask=attention_mask) # BERTへ入力
                    logits = outputs.logits # ミニバッチサイズ×ラベル数のテンソル型の二次元配列．各文章がどのラベルの確率が高いか
                    loss = self.criterion(logits, labels.float())  # ラベルをfloat型に変換

                    total_loss += loss.item() # 誤差加算
                    # 損失関数の勾配を計算
                    loss.backward()
                    # パラメータの最適化実行
                    self.optimizer.step()

                avg_loss = total_loss / len(train_dataloader) # len(train_dataloder)はバッチの数
                # print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

                # 早期終了のチェック
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    current_patience = 0
                else:
                    current_patience += 1
                    if current_patience >= patience:
                        print("Early stopping. No improvement in loss.")
                        break

            return epoch + 1

    # モデルの評価
    def evaluate_model(self, eval_dataloader_1, eval_dataloader_2, model_1, model_2, best_f1, best_model_1, best_model_2, device):
        model_1.eval()
        model_2.eval()
        total_predictions = []
        total_labels = []

        # 予測
        with torch.no_grad():
            for input_ids, attention_mask, labels in eval_dataloader_1:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model_1(input_ids, attention_mask)
                # predictions = torch.sigmoid(outputs.logits)  # シグモイド活性化関数の適用
                predictions = outputs.logits
                predictions = (predictions > 0.5).float()  # 0.5を閾値として二値化

                # predictionsが1の時，そのtextデータについてmodel2を用いた予測を行う
                for i, prediction in enumerate(predictions):
                    if prediction == 1:
                        input_ids = input_ids[i].unsqueeze(0)
                        attention_mask = attention_mask[i].unsqueeze(0)
                        outputs = model_2(input_ids, attention_mask)
                        predictions[i] = torch.argmax(outputs.logits, dim=1)


                total_predictions.extend(predictions.detach().cpu().numpy())
                total_labels.extend(labels.detach().cpu().numpy())

        accuracy = accuracy_score(total_labels, total_predictions)
        f1 = f1_score(total_labels, total_predictions, average='macro', zero_division=0)  # マクロ平均F1スコアの計算
        hamming = hamming_loss(total_labels, total_predictions)  # ハムミング損失の計算

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

        return accuracy, f1, hamming, best_f1, best_model, total_predictions, total_labels  # ハムミング損失を返す

    # 交差検証
    # 2つのデータを入力として入れる．一つは二値のラベルがふられたデータ．２つ目は活動なし以外のラベルがふられたデータ
    def cross_validation(self, model_name, data, num_folds, NUM_EPOCHS, device,
                        LEARNING_RATE, BATCH_SIZE, output_dir, MAX_LEN, timestamp, num_labels):
        # 割合を固定したまま，ランダムに被らないようにデータを分ける．
        skf = KFold(n_splits=num_folds, shuffle=True, random_state=2023)
        # skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2023)
        # skf = IterativeStratification(n_splits=num_folds, shuffle=True, random_state=2023)

        fold_scores = []
        fold_f1 = []
        fold_hamming = []
        best_model = None
        best_f1 = 0.0
        finished_epochs = []
        all_predictions = []
        all_labels = []

        # dataの中のmulti_labelsの活動なしの列を削除したデータを作成
        train_multi_labels = [row[:-1] for row in data['multi_labels']]

        for fold, (train_index, eval_index) in enumerate(skf.split(data['texts'])):
            print(f"-----------------Fold: {fold+1}-----------------")

            # 文章分類を行うクラス：BertForSequenceClassification
            model1 = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) # 二値分類を行うモデル
            model2 = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            # モデルをGPUに配置
            model1.to(device)
            model2.to(device)
            # model1で使用するための訓練データと評価データを辞書型で抽出
            train_dataset1 = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['labels'][i] for i in train_index]
            }
            eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['multi_labels'][i] for i in eval_index]
            }
            # ここでデータセットの前処理を行っている．
            train_encodings1 = preprocess_dataset(train_dataset1, self.tokenizer, MAX_LEN)

            # データローダ型に変形．ミニバッチごとのデータに分割
            train_dataloader1 = DataLoader(train_encodings1, BATCH_SIZE, shuffle=True)
            eval_dataloader1 = DataLoader(eval_encodings1, BATCH_SIZE)

            # 学習させ学習が終わるエポックを返す
            finished_epoch = self.train_2_model(train_dataloader1, model1, NUM_EPOCHS, device, LEARNING_RATE)
            finished_epochs.append(finished_epoch)

            # model2で使用するための訓練データと評価データを辞書型で抽出
            # model2で使用するテキストデータを使用
            train_text = [data['texts'][i] for i in train_index]
            train_label = [data['labels'][i] for i in train_index]
            train_multi_label = [data['multi_labels'][i] for i in train_index]
            eval_text = [data['texts'][i] for i in eval_index]
            eval_label = [data['labels'][i] for i in eval_index]
            eval_multi_label = [data['multi_labels'][i] for i in eval_index]
            # 活動ありのテキストデータを使用
            train_text = [train_text[i] for i,value in enumerate(train_label) if value == 1]
            # 活動ありのmulti_labelを使用
            train_multi_label = [train_multi_label[i] for i,value in enumerate(train_label) if value == 1]
            # 活動ありのテキストデータを使用
            eval_text = [eval_text[i] for i,value in enumerate(eval_label) if value == 1]
            # 活動ありのmulti_labelを使用
            eval_multi_label = [eval_multi_label[i] for i,value in enumerate(eval_label) if value == 1]

            # train_index中のmulti_labelsの活動なしの列を削除したデータを作成
            train_dataset2 = {
                'texts': train_text,
                'labels': train_multi_label
            }
            eval_dataset2 = {
                'texts': eval_text,
                'labels': eval_multi_label
            }

            # ここでデータセットの前処理を行っている．
            train_encodings2 = preprocess_multiclass_dataset(train_dataset2, self.tokenizer, MAX_LEN)
            eval_encodings2 = preprocess_multiclass_dataset(eval_dataset2, self.tokenizer, MAX_LEN)
            # データローダ型に変形．ミニバッチごとのデータに分割
            train_dataloader2 = DataLoader(train_encodings2, BATCH_SIZE, shuffle=True)
            eval_dataloader2 = DataLoader(eval_encodings2, BATCH_SIZE)

            # 学習させ学習が終わるエポックを返す
            finished_epoch = self.train_multi_model(train_dataloader2, model2, NUM_EPOCHS, device, LEARNING_RATE)
            finished_epochs.append(finished_epoch)

            # 学習終了
        
            # 評価
            accuracy, f1, hamming, best_f1, best_model, predictions, labels = self.evaluate_model(eval_dataloader, model1, best_f1, best_model, device)
            fold_scores.append(accuracy)
            fold_f1.append(f1)
            fold_hamming.append(hamming)
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            print(f"Accuracy: {accuracy}")
            print(f"macro f1: {f1}")
            print(f"humming: {hamming}")

        average_accuracy = sum(fold_scores) / num_folds
        average_f1 = sum(fold_f1) / num_folds
        average_hamming = sum(fold_hamming) /num_folds

        print("-------------------------------------")
        print(f"finished epochs : {finished_epochs}")
        print(f"Average Accuracy: {average_accuracy}")
        print(f"Average macro f1: {average_f1}")
        print(f'Average humming: {average_hamming}')

        # モデルの保存
        best_model.save_pretrained(f"{output_dir}/{model_name}{timestamp}")

        return fold_scores, fold_f1, fold_hamming, best_model, all_predictions, all_labels  # 予測値と実際のラベルを返す
    




'''
class MultiClassActClassifier:
    def __init__(self, label_num, model_name, tokenizer, criterion, device, seed=2023):
        set_seed(seed)
        self.label_num = label_num
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.device = device
        self.optimizer = None

    def binary_train_model(self, train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE, patience=3):
        model.train()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        best_loss = float('inf')
        current_patience = 0

        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            epoch_predictions = []

            for step, batch in enumerate(train_dataloader):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                self.optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                _, predictions = torch.max(logits, dim=1)
                epoch_predictions.extend(predictions.detach().cpu().numpy())

            avg_loss = total_loss / len(train_dataloader)

            # 早期終了のチェック
            if avg_loss < best_loss:
                best_loss = avg_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print("Early stopping. No improvement in loss.")
                    break

        return epoch + 1, epoch_predictions

    def binary_evaluate_model(self, eval_dataloader, model, best_f1, best_model, device):
        model.eval()
        total_predictions = []
        total_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in eval_dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask)
                _, predictions = torch.max(outputs.logits, dim=1)

                total_predictions.extend(predictions.detach().cpu().numpy())
                total_labels.extend(labels.detach().cpu().numpy())

        accuracy = accuracy_score(total_labels, total_predictions)
        f1 = f1_score(total_labels, total_predictions, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

        return accuracy, f1, best_f1, best_model, total_predictions

    def multi_train_model(self, train_dataloader, model, NUM_EPOCHS, device, LEARNING_RATE, patience=3):
        model.train()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.BCEWithLogitsLoss()
        best_loss = float('inf')
        current_patience = 0

        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                self.optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(train_dataloader)

            # 早期終了のチェック
            if avg_loss < best_loss:
                best_loss = avg_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print("Early stopping. No improvement in loss.")
                    break

        return epoch + 1

    def multi_evaluate_model(self, eval_dataloader, model, best_f1, best_model, device):
        model.eval()
        total_predictions = []
        total_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in eval_dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask)
                _, predictions = torch.max(outputs.logits, dim=1)

                total_predictions.extend(predictions.detach().cpu().numpy())
                total_labels.extend(labels.detach().cpu().numpy())

        accuracy = accuracy_score(total_labels, total_predictions)
        f1 = f1_score(total_labels, total_predictions, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

        return accuracy, f1, best_f1, best_model

    def cross_validation(self, model_name, data, num_folds, NUM_EPOCHS, device,
                         LEARNING_RATE, BATCH_SIZE, output_dir, MAX_LEN, timestamp):
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2023)

        binary_fold_scores = []
        binary_fold_f1 = []
        binary_best_model = None
        binary_best_f1 = 0.0
        multi_fold_scores = []
        multi_fold_f1 = []
        multi_best_model = None
        multi_best_f1 = 0.0
        finished_epochs = []

        for fold, (train_index, eval_index) in enumerate(skf.split(data['texts'], data['labels'])):
            print(f"-----------------Fold: {fold+1}-----------------")

            binary_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
            multi_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.label_num)

            # 実際のデータセットと同じ形
            binary_train_dataset = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['labels'][i] for i in train_index]
            }
            binary_eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['labels'][i] for i in eval_index]
            }

            binary_train_encodings = preprocess_dataset(binary_train_dataset, self.tokenizer, MAX_LEN)
            binary_eval_encodings = preprocess_dataset(binary_eval_dataset, self.tokenizer, MAX_LEN)

            binary_train_dataloader = DataLoader(binary_train_encodings, BATCH_SIZE, shuffle=True)
            binary_eval_dataloader = DataLoader(binary_eval_encodings, BATCH_SIZE)

            binary_model.to(device)
            multi_model.to(device)

            # 二値分類のトレーニングと評価
            binary_epochs, binary_train_predictions = self.binary_train_model(binary_train_dataloader, binary_model, NUM_EPOCHS, device, LEARNING_RATE)
            accuracy, f1, binary_best_f1, binary_best_model, binary_predictions = self.binary_evaluate_model(binary_eval_dataloader, binary_model, binary_best_f1, binary_best_model, device)
            binary_fold_scores.append(accuracy)
            binary_fold_f1.append(f1)
            finished_epochs.append(binary_epochs)

            multi_train_dataset = {
                'texts': [data['texts'][i] for i in train_index],
                'labels': [data['multi_labels'][i] for i in train_index]
            }
            multi_eval_dataset = {
                'texts': [data['texts'][i] for i in eval_index],
                'labels': [data['multi_labels'][i] for i in eval_index]
            }
            # total_predictionsが1のみのデータセットを作成
            filtered_train_dataset = {
                'texts': [],
                'labels': []
            }
            filtered_eval_dataset = {
                'texts': [],
                'labels': []
            }
            print(len(binary_train_predictions))
            print(len(multi_train_dataset["texts"]))
            print(len(binary_predictions))
            for i, prediction in enumerate(binary_train_predictions):
                if prediction == 1:
                    filtered_train_dataset['texts'].append(multi_train_dataset['texts'][i])
                    filtered_train_dataset['labels'].append(multi_train_dataset['labels'][i])

            for i, prediction in enumerate(binary_predictions):
                if prediction == 1:
                    filtered_eval_dataset['texts'].append(multi_eval_dataset['texts'][i])
                    filtered_eval_dataset['labels'].append(multi_eval_dataset['labels'][i])

            multi_train_encodings = multi_preprocess_dataset(filtered_train_dataset, self.tokenizer, MAX_LEN)
            multi_eval_encodings = multi_preprocess_dataset(filtered_eval_dataset, self.tokenizer, MAX_LEN)

            multi_train_dataloader = DataLoader(multi_train_encodings, BATCH_SIZE, shuffle=True)
            multi_eval_dataloader = DataLoader(multi_eval_encodings, BATCH_SIZE, shuffle=True)

            # 他クラス分類のトレーニングと評価
            multi_epochs = self.multi_train_model(multi_train_dataloader, multi_model, NUM_EPOCHS, device, LEARNING_RATE)
            accuracy, f1, multi_best_f1, multi_best_model = self.multi_evaluate_model(multi_eval_dataloader, multi_model, multi_best_f1, multi_best_model, device)
            multi_fold_scores.append(accuracy)
            multi_fold_f1.append(f1)
            finished_epochs.append(multi_epochs)

        binary_avg_accuracy = sum(binary_fold_scores) / num_folds
        binary_avg_f1 = sum(binary_fold_f1) / num_folds
        multi_avg_accuracy = sum(multi_fold_scores) / num_folds
        multi_avg_f1 = sum(multi_fold_f1) / num_folds

        print("-------------Binary Classification Results-------------")
        print("Average Accuracy:", binary_avg_accuracy)
        print("Average F1 Score:", binary_avg_f1)

        print("-------------Multi-Class Classification Results-------------")
        print("Average Accuracy:", multi_avg_accuracy)
        print("Average F1 Score:", multi_avg_f1)

        # モデルの保存
        binary_best_model.save_pretrained(f"{output_dir}/binary_{model_name}{timestamp}")
        multi_best_model.save_pretrained(f"{output_dir}/multi_{model_name}{timestamp}")

        return binary_avg_accuracy, binary_avg_f1, multi_avg_accuracy, multi_avg_f1, finished_epochs

'''