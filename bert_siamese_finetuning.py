# -*- coding: utf-8 -*-
# @Time    : 2022/1/4 21:23
# @Author  : huangkai
# @File    : bert_siamese_finetuning.py
from bert4keras.backend import keras,  K
from bert4keras.models import build_transformer_model, Model
from keras.models import load_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from keras.layers import merge, LSTM, Bidirectional, Lambda, Input
import numpy as np
from bert4keras.snippets import sequence_padding, DataGenerator


def cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return 1 - merge.dot([x, y], axes=1, normalize=True)


def dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    squared_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))

    return K.mean(y_true * squared_pred + (1 - y_true) * margin_square)


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1,text2, label = l.strip().split('\t')
            D.append((text1,text2, int(label)))
    return D


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = bert_siamese_model.predict(x_true)
        pred = y_pred.ravel() < 0.5
        right += np.sum(pred == y_true.reshape(1, -1))
        total += len(y_true)
    return right / total


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            bert_siamese_model.save('best_model.h5')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False, ):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            batch_token_id, batch_segment_id = [], []
            token_ids, segment_ids = tokenizer.encode(text1, maxlen=maxlen)
            batch_token_id.append(token_ids)
            batch_segment_id.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(text2, maxlen=maxlen)
            batch_token_id.append(token_ids)
            batch_segment_id.append(segment_ids)
            batch_token_ids.append(batch_token_id)
            batch_segment_ids.append(batch_segment_id)
            batch_labels.append([float(label)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = np.array([sequence_padding(k, length=maxlen) for k in batch_token_ids])
                batch_segment_ids = np.array([sequence_padding(k, length=maxlen) for k in batch_segment_ids])
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def bert(model_name="bert"):
    config_path = bert_path + '/bert_config.json'
    checkpoint_path = bert_path + '/bert_model.ckpt'

    bert_model = build_transformer_model(config_path, checkpoint_path, model=model_name)
    return bert_model


def BertSiameseModel(bert_model, use_bilstm=False):
    input_tokens_id = Input(shape=(2, maxlen))
    input_segments_id = Input(shape=(2, maxlen))
    # 切片
    x_in1 = bert_model([input_tokens_id[:, 0, :], input_segments_id[:, 0, :]])
    x_in2 = bert_model([input_tokens_id[:, 1, :], input_segments_id[:, 1, :]])

    if use_bilstm:
        shared_lstm1 = Bidirectional(LSTM(lstm_num, return_sequences=True))
        shared_lstm2 = Bidirectional(LSTM(lstm_num))
    else:
        shared_lstm1 = LSTM(lstm_num, return_sequences=True)
        shared_lstm2 = LSTM(lstm_num)
    q1 = shared_lstm1(x_in1)
    q1 = shared_lstm2(q1)
    q2 = shared_lstm1(x_in2)
    q2 = shared_lstm2(q2)

    distance = Lambda(cosine_distance, output_shape=dist_output_shape)([q1, q2])

    AdamLR = extend_with_piecewise_linear_lr(Adam, name="AdamLR")

    model = Model(inputs=[input_tokens_id, input_segments_id], outputs=distance)
    # model.summary()
    model.compile(loss=contrastive_loss,
                  optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}),
                  metrics=[accuracy])
    return model


if __name__ == '__main__':
    print("read data")
    train_data = load_data("LCQMC/train")
    valid_data = load_data("LCQMC/dev")
    test_data = load_data("LCQMC/test")
    print("global params")
    lstm_num = 100
    maxlen = 32
    batch_size = 128
    bert_path = r"D:\work\pretrained_model\uncased_L-12_H-768_A-12"
    dict_path = bert_path + '/vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    print(" data generator ")
    # data generator
    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size)
    print("build model ")
    evaluator = Evaluator()
    bert_model = bert(model_name="bert")
    bert_siamese_model = BertSiameseModel(bert_model, use_bilstm=True)
    bert_siamese_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )
    bert_siamese_model = bert_siamese_model.load_weights('best_model.h5')
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))

