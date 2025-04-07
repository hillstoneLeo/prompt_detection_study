import joblib
import snoop
import torch
import polars as pl
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
from snoop import pp
from transformers import BertTokenizer, BertModel
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def pred_metrics(label, predict):
    """
    计算预测指标，包括 FPR、FNR、准确率、精确率、召回率和 F1 分数。

    :param label: 真实标签
    :param predict: 预测结果
    :return: 包含各项指标的字典
    """
    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(label, predict)

    return {
        'fpr': fpr[1],  # 误报率
        'fnr': 1 - tpr[1],  # 漏报率
        'accuracy': accuracy_score(label, predict),  # 准确率
        'precision': precision_score(label, predict),  # 精确率
        'recall': recall_score(label, predict),  # 召回率
        'f1-score': f1_score(label, predict)  # F1 分数
    }


def load_bert(model_path: Path = Path('./models/bert')):
    if model_path.exists():
        btok = BertTokenizer.from_pretrained(model_path)
        bmod = BertModel.from_pretrained(model_path)
    else:
        btok = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        btok.save_pretrained(model_path)
        bmod = BertModel.from_pretrained('bert-base-multilingual-uncased')
        bmod.save_pretrained(model_path)
    return btok, bmod


def gen_sentence_embedding(tokenizer, bert_model, prompt):
    """
    使用 BERT 模型生成句子嵌入向量。

    :param tokenizer: 分词器
    :param bert_model: BERT 模型
    :param prompt: 输入文本
    :return: 句子嵌入向量
    """
    # 对输入文本进行分词并转换为张量
    tokens = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
    
    # 在无梯度模式下运行模型
    with torch.no_grad():
        output = bert_model(**tokens)
    
    # 提取最后一层的隐藏状态，并计算平均值作为句子嵌入
    sentence_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return sentence_embedding


@snoop
def load_data(root_folder):
    """
    加载训练和测试数据，并生成嵌入向量。

    :param root_folder: 数据根目录
    :return: 包含训练和测试数据的字典
    """
    # 加载数据集
    ds_train = pl.read_parquet(f'{root_folder}/deepset/data/train-00000-of-00001-9564e8b05b4757ab.parquet')
    ds_test = pl.read_parquet(f'{root_folder}/deepset/data/test-00000-of-00001-701d16158af87368.parquet')
    sf_train = pl.read_parquet(f'{root_folder}/safe-guard/data/train-00000-of-00001.parquet')
    sf_test = pl.read_parquet(f'{root_folder}/safe-guard/data/test-00000-of-00001.parquet')
    
    # 合并训练和测试数据
    combined_train = pl.concat([ds_train, sf_train])
    combined_test = pl.concat([ds_test, sf_test])
    
    # 生成训练和测试数据的嵌入向量
    btok, bmod = load_bert()
    x_train_embeddings = combined_train['text'].map_elements(lambda p: gen_sentence_embedding(btok, bmod, p)).to_list()
    x_test_embeddings = combined_test['text'].map_elements(lambda p: gen_sentence_embedding(btok, bmod, p)).to_list()
    
    # 提取标签
    y_train = combined_train['label']
    y_test = combined_test['label']
    
    # 返回数据字典
    return {
        'x-train': pl.from_records(x_train_embeddings, orient='row'),
        'x-test': pl.from_records(x_test_embeddings, orient='row'),
        'y-train': y_train,
        'y-test': y_test
    }


@snoop
def train_models(data_path: str, model_path: str) -> pl.DataFrame:
    """
    训练多个模型，并返回每个模型的评估指标。

    :param data_path: 数据文件路径
    :param model_path: 模型保存路径
    :return: 包含模型评估指标的 DataFrame
    """
    # 加载数据
    data = load_data(data_path)
    
    # 定义模型路径
    model_base_path = Path(model_path)

    # 定义模型字典
    estimators = {
        "naive-bayes": GaussianNB(),
        "logistic-regression": LogisticRegression(),
        "support-vector-machine": SVC(),
        "random-forest": RandomForestClassifier()
    }

    # 用于存储评估结果的列表
    evaluation_results = []

    # 遍历每个模型并进行训练和评估
    for estimator_name, model in estimators.items():
        # 提取训练和测试数据
        X_train = data["x-train"]
        y_train = data["y-train"]
        X_test = data["x-test"]
        y_test = data["y-test"]

        # 训练模型
        model.fit(X_train, y_train)

        # 保存模型
        model_file_path = model_base_path.joinpath(f"{estimator_name}.pkl")
        joblib.dump(model, model_file_path)

        # 进行预测
        y_pred = model.predict(X_test)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 将结果存储到字典中
        result = {
            "Estimator": estimator_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        # 将字典添加到结果列表中
        evaluation_results.append(result)

    return pl.from_dicts(evaluation_results)


@snoop
def predict_injection(data_path: str, model_name: str):
    """
    使用指定模型对数据进行预测，并计算预测指标。
    
    :param data_path: 数据文件路径
    :param model_name: 模型名称
    :return: 预测指标
    """
    # 加载模型
    model = joblib.load(f'./models/{model_name}.pkl')

    # 读取数据
    test_data = pl.read_csv(data_path, separator='\t')

    # 提取标签列
    y_test = test_data['label']

    # 生成文本嵌入向量
    btok, bmod = load_bert()
    text_embeddings = test_data['text'].map_elements(lambda p: gen_sentence_embedding(btok, bmod, p)).to_list()

    # 对嵌入向量进行预测
    y_pred = model.predict(pl.from_records(text_embeddings, orient='row'))

    # 将预测结果添加到数据框中
    predict_results = test_data.with_columns(predict=y_pred)

    # 将预测结果保存到文件
    predict_results.write_csv('injection-predict.tsv', separator='\t')

    # 计算预测指标
    return pred_metrics(y_test, y_pred)


pp(train_models('./data', './models'))
pp(predict_injection("./data/complete-questions.tsv", "svm"))
