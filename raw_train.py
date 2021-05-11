#coding=utf-8

import csv
import os

import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.optimizers import Adam, extend_with_gradient_accumulation, extend_with_piecewise_linear_lr
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
from sklearn.model_selection import train_test_split
from bert4keras.layers import Loss, Layer
from sklearn.metrics import f1_score
import random
import json
from tqdm import tqdm
import pickle
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
K.clear_session()  # 清空当前 session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
K.set_session(sess)

set_gelu('tanh')  # 切换gelu版本

maxlen = 128
batch_size = 32



config_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

train_data_dir = './souhu_data'
save_model_dir = './output'

test_pth = {
    'task_A': './souhu_data/data_A_test.json',
    'task_B': './souhu_data/data_B_test.json',
}
train_pth = {
    'task_A': './souhu_data/data_A_train.json',
    'task_B': './souhu_data/data_B_train.json'
}

valid_pth = {
    'task_A': './souhu_data/data_A_val.json',
    'task_B': './souhu_data/data_B_val.json'
}

label_map = {'0': 0, 0: '0', 'num': 2, '1': 1, 1: '1'}

predict_dir = f'{save_model_dir}/predict_result'



token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class MTL_train_data_generator(object):
    def __init__(self, train_data_dir, tokenizer, batch_size=16, maxlen=128):
        self.class_A_train = json.load(fp=open(os.path.join(train_data_dir, 'data_A_train.json'), 'r', encoding='utf-8'))
        self.class_B_train = json.load(fp=open(os.path.join(train_data_dir, 'data_B_train.json'), 'r', encoding='utf-8'))
        self.batch_size = batch_size
        self.max_len = maxlen
        self.tokenizer = tokenizer

    def __iter__(self):
        class_A_index = 0
        class_B_index = 0
        loop = {
            'class_A': 'class_B',
            'class_B': 'class_A',
        }
        class_A_end = False
        class_B_end = False

        cur_state = 'class_A'

        random.shuffle(self.class_A_train)
        random.shuffle(self.class_B_train)

        while (not class_A_end or class_B_end):
            if 'class_A' in cur_state:
                # 索引加上当前的batch_size大于训练集的长度的话就代表此类训练集应该结束
                if class_A_index + self.batch_size > len(self.class_A_train):
                    class_A_end = True
                    cur_state = loop[cur_state]
                else:
                    # 定义一个batch大小的训练集
                    batch_data = self.class_A_train[class_A_index: class_A_index + self.batch_size]
                    batch_token_ids = []
                    batch_segment_ids = []
                    # item以字典的形式存储
                    for item in batch_data:
                        token_ids, segment_ids = self.tokenizer.encode(item["source"], item["target"], maxlen=self.max_len)
                        batch_token_ids.append(token_ids)
                        batch_segment_ids.append(segment_ids)
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = np.array([[item["label_A"]] for item in batch_data], dtype=np.int32)
                    batch_task_ids = np.ones((self.batch_size, 1), dtype=np.int32) * 1   # 任务A的id是1


                    yield [batch_token_ids, batch_segment_ids, batch_labels, batch_task_ids], None
                    cur_state = loop[cur_state]
                    class_A_index += self.batch_size

            if 'class_B' in cur_state:
                # 索引加上当前的batch_size大于训练集的长度的话就代表此类训练集应该结束
                if class_B_index + self.batch_size > len(self.class_B_train):
                    class_B_end = True
                    cur_state = loop[cur_state]
                else:
                    # 定义一个batch大小的训练集
                    batch_data = self.class_B_train[class_B_index: class_B_index + self.batch_size]
                    batch_token_ids = []
                    batch_segment_ids = []
                    # item以字典的形式存储
                    for item in batch_data:
                        token_ids, segment_ids = self.tokenizer.encode(item["source"], item["target"], maxlen=self.max_len)
                        batch_token_ids.append(token_ids)
                        batch_segment_ids.append(segment_ids)
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = np.array([[item["label_B"]] for item in batch_data], dtype=np.int32)
                    batch_task_ids = np.ones((self.batch_size, 1), dtype=np.int32) * 2  # 任务B的id是2

                    yield [batch_token_ids, batch_segment_ids, batch_labels, batch_task_ids], None
                    cur_state = loop[cur_state]
                    class_B_index += self.batch_size

    def __len__(self):
        min_mun = min([len(self.class_A_train), len(self.class_B_train)])
        return int(min_mun / batch_size) * 2

    def forfit(self):
        while True:
            # 数据生成
            for d in self.__iter__():
                yield d

# def load_train_data(train_data_dir):
#     CLASS_A_train = json.load(fp=open(os.path.join(train_data_dir, 'data_A_train.json'), 'r', encoding='utf-8'))
#     CLASS_B_train = json.load(fp=open(os.path.join(train_data_dir, 'data_B_train.json'), 'r', encoding='utf-8'))
#
#     return CLASS_A_train + CLASS_B_train




class MTl_val_data_generator(DataGenerator):
    # 数据生成器
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels, batch_task_ids = [], [], [], []
        for is_end, D in self.sample(random):
            task_id = D["task_id"]
            if task_id == 1:
                text1 = D["source"]
                text2 = D["target"]
                label = D["label_A"]
                token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            else:
                text1 = D["source"]
                text2 = D["target"]
                label = D["label_B"]
                token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            batch_task_ids.append([task_id])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = np.array(batch_labels)
                batch_task_ids = np.array(batch_task_ids)
                yield [batch_token_ids, batch_segment_ids, batch_labels, batch_task_ids], None
                batch_token_ids, batch_segment_ids, batch_labels, batch_task_ids = [], [], [], []

def load_dev_data(train_data_dir):
    CLASS_A_dev = json.load(fp=open(os.path.join(train_data_dir, 'data_A_val.json'), 'r', encoding='utf-8'))
    CLASS_B_dev = json.load(fp=open(os.path.join(train_data_dir, 'data_B_val.json'), 'r', encoding='utf-8'))

    return CLASS_A_dev + CLASS_B_dev

train_loader = MTL_train_data_generator(train_data_dir, tokenizer, batch_size=batch_size, maxlen=maxlen)
dev_loader = MTl_val_data_generator(load_dev_data(train_data_dir), batch_size=batch_size)


# 加载预训练模型
# bert = build_transformer_model(
#     config_path=config_path,
#     checkpoint_path=checkpoint_path,
#     with_pool=True,
#     return_keras_model=False,
# )

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)  # bert4keras中的models.py里面的build_transformer_model方法

output = Dropout(rate=0.1)(bert.model.output[0])

output_A = Dense(
    units=2, activation='softmax', kernel_initializer=bert.initializer, name='class_A_output'
)(output)

output_B = Dense(
    units=2, activation='softmax', kernel_initializer=bert.initializer, name='class_B_output'
)(output)

I_labels = keras.layers.Input(shape=(1,), dtype='int32', name='labels')
I_task_ids = keras.layers.Input(shape=(1,), dtype='int32', name='task_ids')

model = keras.models.Model(bert.model.input + [I_labels, I_task_ids], [output_A, output_B])


class CrossEntropy(Loss):
    def __init(self, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)
        pass

    def compute_loss(self, inputs, mask=None):
        _, _, labels, task_ids, output0, output1 = inputs
        # K.squeeze去掉维数为1的维度，如原来的是[1,2,3,4,1,1]，则会变为[2,3]
        labels = K.squeeze(labels, axis=-1)
        task_ids = K.squeeze(task_ids, axis=-1)

        labels0 = labels[K.equal(task_ids, 1)]
        labels1 = labels[K.equal(task_ids, 2)]

        output0_0 = output0[K.equal(task_ids, 1)]
        output1_1 = output1[K.equal(task_ids, 2)]
        """
        tf.cond()类似于if 和 else，如果第一句话的条件为真则执行第二句话，否则执行第三句话
        如果tf.equal(K.shape(output0_0)[0], 0)为真，就执行lambda : K.reshape(output0_0, (-1, )), 
        否则执行lambda: K.sparse_categorical_crossentropy(labels0, output0_0)
        """

        loss0 = tf.cond(tf.equal(K.shape(output0_0)[0], 0), lambda : K.reshape(output0_0, (-1, )), lambda: K.sparse_categorical_crossentropy(labels0, output0_0))
        loss1 = tf.cond(tf.equal(K.shape(output1_1)[0], 0), lambda : K.reshape(output1_1, (-1, )), lambda: K.sparse_categorical_crossentropy(labels1, output1_1))

        loss = K.mean(K.concatenate([loss0, loss1], axis=0))

        return loss

output = CrossEntropy(output_axis=[4, 5])(model.input+model.output)

model = keras.models.Model(model.inputs, output)



optimizer = extend_with_piecewise_linear_lr(Adam)
optimizer = extend_with_gradient_accumulation(optimizer)

optimizer_params = {
        'learning_rate': 2e-5,
        'lr_schedule': {0: 1.0, len(train_loader)*2: 0.5},
        'grad_accum_steps': 8,
    }
optimizer = optimizer(**optimizer_params)
model.compile(
    optimizer=optimizer,  # 用足够小的学习率
)
model.summary()

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, dev_loader, save_model_dir):
        self.dev_loader = dev_loader
        self.best_total_f1_score = -999
        self.save_model_dir = save_model_dir
        self.cur_batch = 0


    def on_epoch_end(self, epoch, logs=None):
        total_f1_score, CLASS_A_f1, CLASS_B_f1 = self.evaluate()
        if total_f1_score > self.best_total_f1_score:
            print(f'total_f1_score improve {self.best_total_f1_score} -> {total_f1_score}   !!! save best modle!')
            self.best_total_f1_score = total_f1_score
            self.model.save_weights(os.path.join(save_model_dir, 'best_model.weights'))
        print(f'\nclass_A f1_score: {CLASS_A_f1}')
        print(f'\nclass_B f1_score: {CLASS_B_f1}')
        print(f'total_f1_score: {total_f1_score}')

    def evaluate(self):
        CLASS_A_pred = []
        CLASS_B_pred = []
        CLASS_A_label = []
        CLASS_B_label = []
        for t, _ in tqdm(self.dev_loader):
            batch = 0
            y_true = t[2].squeeze(-1)
            task_ids = t[3].squeeze(-1)
            outA_pred, outB_pred = self.model.predict(t)
            for id in task_ids:
                if id == 1:
                    CLASS_A_pred.append(outA_pred[batch].argmax())
                    CLASS_A_label.append(y_true[batch])
                else:
                    CLASS_B_pred.append(outA_pred[batch].argmax())
                    CLASS_B_label.append(y_true[batch])
                batch += 1

        CLASS_A_f1_score = f1_score(CLASS_A_label, CLASS_A_pred, average='macro')
        CLASS_B_f1_score = f1_score(CLASS_B_label, CLASS_B_pred, average='macro')
        total_f1_score = (CLASS_A_f1_score + CLASS_B_f1_score) / 2
        return total_f1_score, CLASS_A_f1_score, CLASS_B_f1_score




class MTL_test_data_generator(DataGenerator):
    '''
    数据生成器
    '''
    def __iter__(self, random=False):
        batch_token_ID, batch_token_ids, batch_segment_ids, batch_task_ids = [], [], [], []
        for is_end, D in self.sample(random):
            task_id = D["task_id"]
            if task_id == 1:
                text1 = D["text1"]
                text2 = D["text2"]
                token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            else:
                text1 = D["text1"]
                text2 = D["text2"]
                token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)

            batch_token_ID.append(D["id"])
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_task_ids.append([task_id])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_task_ids = np.array(batch_task_ids)
                yield batch_token_ID, [batch_token_ids, batch_segment_ids, batch_task_ids, batch_task_ids]
                batch_token_ID, batch_token_ids, batch_segment_ids, batch_task_ids = [], [], [], []


def test(model):
    def load_data_test(pth):

        data = []
        f = open(pth, encoding='utf-8')
        setting = json.load(f)
        for i in setting:
            if 'data_A_test' in pth:
                text1 = i["source"]
                text2 = i["target"]
                _id = i["id_"]
                task_id = i["task_id"]
                D = {
                    "id": _id,
                    "text1": text1.strip(),
                    "text2": text2.strip(),
                    "task_id": task_id,
                }
            else:
                text1 = i["source"]
                text2 = i["target"]
                _id = i["id_"]
                task_id = i["task_id"]
                D = {
                    "id": _id,
                    "text1": text1.strip(),
                    "text2": text2.strip(),
                    "task_id": task_id,
                }
            data.append(D)
        return data

    with open(os.path.join(predict_dir, 'predict.csv'), 'w', encoding='utf-8') as fp:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id", "label"])

    for data_type, pth in test_pth.items():
        print(f'predict {data_type}')
        data = load_data_test(pth)
        test_generator = MTL_test_data_generator(data, batch_size=batch_size)

        IDs = []
        pred_label = []

        for batch_token_ID, t in tqdm(test_generator):
            batch = 0
            task_ids = t[3].squeeze(-1)
            out1_pred, out2_pred = model.predict(t)
            for id in task_ids:
                if id == 1:
                    pred_label.append(label_map[out1_pred[batch].argmax()])
                else:
                    pred_label.append(label_map[out1_pred[batch].argmax()])
                IDs.append(batch-token_ID[batch])
                batch += 1

            for index, _id in enumerate(IDs):
                csv_writer.writerow([_id, pred_label[index]])






if __name__ == '__main__':

    evaluator = Evaluator(dev_loader, save_model_dir)

    model.fit(
        train_loader.forfit(),
        steps_per_epoch=len(train_loader),
        epochs=5,
        callbacks=[evaluator],
    )

    print(f'restore the best model from {os.path.join(save_model_dir, "best_model.weights")}')
    model.load_weights(os.path.join(save_model_dir, 'best_model.weights'))
    test(model)

    # model.load_weights('best_model.weights')
    # # print(u'final test acc: %05f\n' % (evaluate(test_generator)))
    # test(model)

# else:
#
#     model.load_weights('best_model.weights')