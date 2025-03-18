import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

# ラベルをマージする関数
def load_and_merge_labels(DATASET_PATH, merge_labels=None):
    """
    データセットの読み込み
    merge_labels: マージしたいラベルの辞書。例: {'C': 'A'} は 'C' を 'A' にマージする
    """
    df = pd.read_excel(DATASET_PATH)

    # テキストデータとラベルデータを取得する
    texts = df['text'].values.tolist()
    
    # ラベルデータをリストのリストとして取得
    labels = df[['label1', 'label2', 'label3', 'label4', 'label5']].fillna('').values.tolist()
    
    # ラベルのマージ処理
    if merge_labels:
        for row in labels:
            for i, label in enumerate(row):
                if label in merge_labels:
                    row[i] = merge_labels[label]

    # ラベルデータをワンホットエンコーディングに変換
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    class_name = mlb.classes_[1:]
    labels_list = labels.tolist()
    labels_list = [row[1:] for row in labels_list] # ０列目の空白を削除
    
    # テキストとラベルを含む辞書型のデータセットとして返す
    dataset = {
        'texts': texts,
        'labels': labels_list
    }

    return dataset, class_name


# データセットの読み込み.dfからテキストとラベルを含めたデータセットを返す
def load_dataset_2class_classification(DATASET_PATH):
    """データセットの読み込み"""
    df = pd.read_excel(DATASET_PATH)

    # テキストデータとラベルデータを取得する
    df["labels"] = 1
    df.loc[df["label1"] == "4. 行動なし", "labels"] = 0
    texts = df['text'].values.tolist()
    labels = df['labels'].values.tolist()

    # テキストとラベルを含む辞書型のデータセットとして返す
    dataset = {
        'texts': texts,
        'labels': labels
    }

    return dataset

# データセットの読み込み.dfからテキストを含めるデータセットを返す
# データセットはcsvファイルに限る
def load_text_dataset(DATASET_PATH):
    """データセットの読み込み"""
    df = pd.read_excel(DATASET_PATH)
    # テキストデータを取得
    texts = df['text'].values.tolist()
    # テキストを含む辞書型のデータセットとして返す
    dataset = {
        'texts': texts
    }
    return dataset

def load_sequential_emotion_data(DATASET_PATH):
    """感情データを列の順番で結合してリストとして読み込む"""
    df = pd.read_excel(DATASET_PATH)
    
    # 必要な列名を指定
    columns = ['Sadness', 'Anxiety', 'Anger', 'Disgust', 'Trust', 'Surprise', 'Joy']
    
    # 列ごとにデータを結合
    combined_texts = []
    for column in columns:
        combined_texts.extend(df[column].values.tolist())

    # 結合されたデータを含む辞書を返す
    dataset = {
        'texts': combined_texts
    }
    return dataset

# データパスからテキストとラベルを含めたデータセット，クラス名を返す
def load_multiclass_dataset(DATASET_PATH):
    """データセットの読み込み"""
    df = pd.read_excel(DATASET_PATH)

    # テキストデータとラベルデータを取得する
    texts = df['text'].values.tolist()
    
    # ラベルデータをリストのリストとして取得
    labels = df[['label1', 'label2', 'label3', 'label4', 'label5']].fillna('').values.tolist()
    
    # ラベルデータをワンホットエンコーディングに変換
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    class_name = mlb.classes_[1:]
    labels_list = labels.tolist()
    labels_list = [row[1:] for row in labels_list] # ０列目の空白を削除
    # テキストとラベルを含む辞書型のデータセットとして返す
    dataset = {
        'texts': texts,
        'labels': labels_list
    }

    return dataset, class_name

def load_2class_multiclass_dataset(Dataset_PATH):
    dataset1 = load_dataset(Dataset_PATH)
    dataset2, class_name = load_multiclass_dataset(Dataset_PATH)
    dataset = {
        'texts': dataset1['texts'],
        'labels': dataset1['labels'],
        'multi_labels': dataset2['labels']
    }
    return dataset, class_name

def load_dataset_2class_classification_v2(DATASET_PATH):
    """データセットの読み込み"""
    df = pd.read_excel(DATASET_PATH)
    # テキストデータとラベルデータを取得する
    texts = df['text'].values.tolist()
    # ラベルデータをリストのリストとして取得
    labels = df[['label1', 'label2', 'label3', 'label4', 'label5']].fillna('').values.tolist()
    # ラベルの最初の数字を取得して新しいラベルとして使用
    new_labels = []
    for label_list in labels:
        new_label_list = [label.split(' ')[0][0] if label else '' for label in label_list]
        new_labels.append(new_label_list)
    # ラベルデータをワンホットエンコーディングに変換
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(new_labels)
    class_name = np.array(['0','1'])
    labels_list = labels.tolist()
    # labels_listの4列目が1のデータでない時に一列目を1にする2列のリストを作成
    labels_list = [[0 if row[4] == 1 else 1, 1 if row[4] == 1 else 0] for row in labels_list]
    # テキストとラベルを含む辞書型のデータセットとして返す
    dataset = {
        'texts': texts,
        'labels': labels_list
    }
    return dataset, class_name

def load_dataset_4class_Multi_classification(DATASET_PATH):
    """データセットの読み込み"""
    df = pd.read_excel(DATASET_PATH)

    # テキストデータとラベルデータを取得する
    texts = df['text'].values.tolist()
    
    # ラベルデータをリストのリストとして取得
    labels = df[['label1', 'label2', 'label3', 'label4', 'label5']].fillna('').values.tolist()
    
    # ラベルの最初の数字を取得して新しいラベルとして使用
    new_labels = []
    for label_list in labels:
        new_label_list = [label.split(' ')[0][0] if label else '' for label in label_list]
        new_labels.append(new_label_list)
    
    # ラベルデータをワンホットエンコーディングに変換
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(new_labels)
    class_name = mlb.classes_[1:]
    labels_list = labels.tolist()
    labels_list = [row[1:] for row in labels_list] # ０列目の空白を削除
    
    # テキストとラベルを含む辞書型のデータセットとして返す
    dataset = {
        'texts': texts,
        'labels': labels_list
    }

    return dataset, class_name

def load_dataset_3class_Multi_classification(DATASET_PATH):
    # データセットの読み込み
    dataset_4class, class_name = load_dataset_4class_Multi_classification(DATASET_PATH)
    # dataset_4class['labels'][-1]が1のデータを削除
    # dataset_3classにdataset_4class['labels'][:-1]を格納
    texts = []
    labels = []
    for i in range(len(dataset_4class['labels'])):
        if dataset_4class['labels'][i][-1] == 0:
            texts.append(dataset_4class['texts'][i])
            labels.append(dataset_4class['labels'][i][:-1])
    dataset_3class = {
        'texts': texts,
        'labels': labels
    }
    class_name = class_name[:-1]
    return dataset_3class, class_name


# def load_multi_dataset(DATASET_PATH):
#     """データセットの読み込み"""
#     df = pd.read_excel(DATASET_PATH)

#     # テキストデータとラベルデータを取得する
#     df["labels"] = 1
#     df.loc[df["new_label"] == "活動なし", "labels"] = 0
#     texts = df['text'].values.tolist()
#     binary_labels = df['labels'].values.tolist()

#     m_labels = []
#     for label_str in df['new_label']:
#         label_list = label_str.split('，')
#         label_list = [label.strip() for label in label_list]
#         m_labels.append(label_list)

#     labels_set = {'メディア': 0,
#                   '交際-オンライン的接触': 1,
#                   '交際-物理的接触': 2,
#                   '仕事': 3,
#                   '喫煙': 4,
#                   '家事': 5,
#                   '睡眠': 6,
#                   '移動': 7,
#                   '買い物': 8,
#                   '趣味・娯楽-体動かさない': 9,
#                   '趣味・娯楽-体動かす': 10,
#                   '趣味・娯楽ー体動かす': 11,
#                   '身の回りの用事': 12,
#                   '食事-飲酒あり': 13,
#                   '食事-飲酒なし': 14,
#                   '活動なし': 15}

#     binary_m_labels = map_multi_labels_to_binary(m_labels, labels_set)

#     # データセットとして返す
#     dataset = {
#         'texts': texts,
#         'labels': binary_labels,
#         'act_labels': m_labels,
#         'multi_labels': binary_m_labels
#     }
#     return dataset

# Sadness, Anxiety, Anger, Disgust, Trust, Surprise, Joyの列があるデータセットを読み込む
def load_emotion_text_dataset(DATASET_PATH):
    """データセットの読み込み"""
    df = pd.read_excel(DATASET_PATH)
    # テキストデータを取得
    Sadness_texts = df['Sadness'].values.tolist()
    Anxiety_texts = df['Anxiety'].values.tolist()
    Anger_texts = df['Anger'].values.tolist()
    Disgust_texts = df['Disgust'].values.tolist()
    Trust_texts = df['Trust'].values.tolist()
    Surprise_texts = df['Surprise'].values.tolist()
    Joy_texts = df['Joy'].values.tolist()

    # テキストを含む辞書型をリストとして返す
    emotion_dataset = [{'texts': Sadness_texts}, {'texts': Anxiety_texts}, {'texts': Anger_texts},
               {'texts': Disgust_texts}, {'texts': Trust_texts}, {'texts': Surprise_texts}, {'texts': Joy_texts}]
    return emotion_dataset

# データのtexts列をトークン化したデータに変換する関数
def tokenize_function(data, tokenizer, max_len):
    return tokenizer(data['texts'], padding='max_length', truncation=True, max_length=max_len)

# datasetをTrainerに入力する形に変形する関数
def preprocess_for_Trainer(dataset, tokenizer, max_len):
    # テキストデータをチェックし、NaNなら空文字に置き換える
    dataset['texts'] = [str(text) if not pd.isna(text) else "" for text in dataset['texts']]
    # データセットをDataset型に変換
    data = Dataset.from_dict(dataset)
    # データセットをトークン化
    data = data.map(tokenize_function,
                    batched=True,
                    fn_kwargs={'tokenizer': tokenizer, 'max_len': max_len})
    return data

# テストデータを分割する関数
def split_test_data(data, test_size, SEED):
    # labelの分布を維持したままデータを分割
    # train_test_splitにより、データとインデックスの両方を取得
    '''
    train_texts, test_texts, train_labels, test_labels, train_indices, test_indices = train_test_split(
        data['texts'], data['labels'], range(len(data['texts'])), test_size=test_size, stratify=data['labels'], random_state=SEED
    )
    '''
    train_texts, test_texts, train_labels, test_labels, train_indices, test_indices = train_test_split(
        data['texts'], data['labels'], range(len(data['texts'])), test_size=test_size, random_state=SEED
    )
    # 分割したデータを辞書型で格納
    train_data = {'texts': train_texts, 'labels': train_labels}
    test_data = {'texts': test_texts, 'labels': test_labels}
    # 分割したデータのインデックスも返す
    return train_data, test_data, train_indices, test_indices

# マルチラベルデータセットの分割関数
def split_multilabel_data(data, test_size, SEED):
    # これで再現性を持たせることができる
    np.random.seed(SEED)
    # マルチラベルデータの分割
    X_train, y_train, X_test, y_test = iterative_train_test_split(np.array(data['texts']).reshape(-1,1), np.array(data['labels']), test_size=test_size)
    # 分割したデータを辞書型で格納．リストに直して格納
    train_data = {'texts': X_train.reshape(-1).tolist(), 'labels': y_train.tolist()}
    test_data = {'texts': X_test.reshape(-1).tolist(), 'labels': y_test.tolist()}
    return train_data, test_data

# 二つの辞書型データがあり，同じ['texts']を持つindexを取得し，そのindexのデータのみの辞書型データを返す関数
def split_same_texts_data(data1, data2):
    # data1['texts']と同じテキストを含むdata2['texts']のインデックスを取得
    index = []
    for i in range(len(data1['texts'])):
        # data1['texts']と同じテキストを含むdata2['texts']のインデックスを取得
        index.append(data2['texts'].index(data1['texts'][i]))
    # indexを使ってdata2のデータを取得
    new_data2 = {'texts': [data2['texts'][i] for i in index], 'labels': [data2['labels'][i] for i in index]}
    return new_data2
 

# テストデータを分割する関数_分割の際にstratifyを使用する
def split_test_data_stratify(data, test_size, SEED):
    # labelの分布を維持したままデータを分割
    # train_test_splitにより、データとインデックスの両方を取得
    train_texts, test_texts, train_labels, test_labels, train_indices, test_indices = train_test_split(
        data['texts'], data['labels'], range(len(data['texts'])), test_size=test_size, stratify=data['labels'], random_state=SEED
    )
    # 分割したデータを辞書型で格納
    train_data = {'texts': train_texts, 'labels': train_labels}
    test_data = {'texts': test_texts, 'labels': test_labels}
    # 分割したデータのインデックスも返す
    return train_data, test_data, train_indices, test_indices

# 辞書をtrain_indexとtest_indexを引数にとり、train_dataとtest_dataを返す関数
def split_data_by_index(data, train_index, test_index):
    # train_indexとtest_indexからtrain_dataとtest_dataを作成
    train_data = {'texts': [data['texts'][i] for i in train_index], 'labels': [data['labels'][i] for i in train_index]}
    test_data = {'texts': [data['texts'][i] for i in test_index], 'labels': [data['labels'][i] for i in test_index]}
    return train_data, test_data

# 入力はtextsのみが含まれている辞書データであり，前処理を終えたデータセットを返す
def preprocess_text_dataset(dataset, tokenizer, MAX_LEN):
    """データセットの前処理とトークン化"""
    encoding = tokenizer.batch_encode_plus(
        dataset['texts'],
        padding='longest', # バッチ内の最長のシーケンスに合わせて他のシーケンスをパディングする
        truncation=True, # トークン化されたシーケンスがMAX_LENを超える場合は切り捨てを行う
        max_length=MAX_LEN,
        return_tensors='pt' # エンコーディングをPyTorchテンソルとして返すことを指定
    )
    input_ids = encoding['input_ids'] # トークンID. データ数×105(105トークンが全体で最大のトークン数だから)
    attention_mask = encoding['attention_mask'] #[PAD]の部分，つまいパディングした部分のみ0でそれ以外1のリスト．データ数×105

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    return dataset


# 入力は辞書型のデータセット
def preprocess_dataset(dataset, tokenizer, MAX_LEN):
    """データセットの前処理とトークン化"""
    # バッチ単位でテキストデータのトークン化とエンコーディングを行う．
    # 複数の文章を符号化するために使われる．一文には使えない．
    encodings = tokenizer.batch_encode_plus(
        dataset['texts'],
        padding='longest', # バッチ内の最長のシーケンスに合わせて他のシーケンスをパディングする
        truncation=True, # トークン化されたシーケンスがMAX_LENを超える場合は切り捨てを行う
        max_length=MAX_LEN,
        return_tensors='pt' # エンコーディングをPyTorchテンソルとして返すことを指定
    )
    input_ids = encodings['input_ids'] # トークンID. データ数×105(105トークンが全体で最大のトークン数だから)
    attention_mask = encodings['attention_mask'] #[PAD]の部分，つまいパディングした部分のみ0でそれ以外1のリスト．データ数×105
    labels = torch.tensor(dataset['labels']) # labelをテンソル化．1次元配列

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    return dataset

def preprocess_multiclass_dataset(dataset, tokenizer, MAX_LEN):
    """データセットの前処理とトークン化"""
    # バッチ単位でテキストデータのトークン化とエンコーディングを行う．
    # 複数の文章を符号化するために使われる．一文には使えない．
    encodings = tokenizer.batch_encode_plus(
        dataset['texts'],
        padding='longest', # バッチ内の最長のシーケンスに合わせて他のシーケンスをパディングする
        truncation=True, # トークン化されたシーケンスがMAX_LENを超える場合は切り捨てを行う
        max_length=MAX_LEN,
        return_tensors='pt' # エンコーディングをPyTorchテンソルとして返すことを指定
    )
    input_ids = encodings['input_ids'] # トークンID. データ数×105(105トークンが全体で最大のトークン数だから)
    attention_mask = encodings['attention_mask'] #[PAD]の部分，つまいパディングした部分のみ0でそれ以外1のリスト．データ数×105
    labels = torch.tensor(dataset['labels'])  # ラベルをテンソル化．2次元配列

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    return dataset

# def multi_preprocess_dataset(dataset, tokenizer, max_len):
#     """データセットの前処理とトークン化"""
#     texts = dataset['texts']
#     labels = dataset['labels']
#     m_labels = dataset['labels']
#     print(labels)

#     encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
#     input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
#     attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)

#     labels_set = {'メディア': 0,
#             '交際-オンライン的接触': 1,
#             '交際-物理的接触': 2,
#             '仕事': 3,
#             '喫煙': 4,
#             '家事': 5,
#             '睡眠': 6,
#             '移動': 7,
#             '買い物': 8,
#             '趣味・娯楽-体動かさない': 9,
#             '趣味・娯楽-体動かす': 10,
#             '趣味・娯楽ー体動かす': 11,
#             '身の回りの用事': 12,
#             '食事-飲酒あり': 13,
#             '食事-飲酒なし': 14,
#             '活動なし': 15}

#     label_ids = []
#     for label in labels:
#         binary = torch.zeros(len(labels_set))
#         if str(label.item()) in labels_set:
#             binary[labels_set[str(label.item())]] = 1
#         label_ids.append(binary)
#     labels = torch.stack(label_ids)

#     dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
#     return dataset

def multi_preprocess_dataset(dataset, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    labels = []

    for data in dataset:
        encoded = tokenizer.encode_plus(
            data['text'],
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids.append(encoded['input_ids'].squeeze())
        attention_masks.append(encoded['attention_mask'].squeeze())
        labels.append(torch.tensor(data['labels']).squeeze())

    return {
        'input_ids': torch.stack(input_ids),
        'attention_masks': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }


def load_sequential_emotion_data(DATASET_PATH):
    """感情データを列の順番で結合してリストとして読み込む"""
    df = pd.read_excel(DATASET_PATH)
    
    # 必要な列名を指定
    columns = ['Sadness', 'Anxiety', 'Anger', 'Disgust', 'Trust', 'Surprise', 'Joy']
    
    # 列ごとにデータを結合
    combined_texts = []
    for column in columns:
        combined_texts.extend(df[column].values.tolist())

    # 結合されたデータを含む辞書を返す
    dataset = {
        'texts': combined_texts
    }
    return dataset
