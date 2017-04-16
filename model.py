# -*- encoding: utf-8 -*-

from keras.layers import Input, Embedding, LSTM, Convolution1D
from keras.layers import MaxPooling1D, AveragePooling1D, Bidirectional
from keras.layers import merge, Dropout, Flatten, Dense, Permute
from keras.models import Model, Sequential
from keras.regularizers import l2


def buildBiLSTM(*args):
	model = Sequential()
	model.add(Embedding(input_dim=args[0],
						output_dim=100,
						weights=[args[1]],
						input_length=args[2]))
	# merge_mode 前向和后向RNN输出的结合方式 : sum , mul , concat , ave 和 None 之一
	# model.add(Bidirectional(LSTM(100), merge_mode='sum'))  # 双向求和--100维
	model.add(Bidirectional(LSTM(100), merge_mode='concat'))  # 双向拼接--200维
	# model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	return model


def build_AC_BLSTM(*args):
	EMBEDDING_DIM = 100
	MAX_SEQUENCE_LENGTH = 10
	labels_num = 2
	NB_FILTER = 100
	FILTER_LENGTH = [2,3,5] # 卷积核的时域长度
	POOL_LENGTH = 10
	# 将输入序列编码成一个指定维度向量组成的序列
	embedding_layer = Embedding(input_dim=args[0],
								output_dim=EMBEDDING_DIM,
								weights=[args[1]],
								input_length=MAX_SEQUENCE_LENGTH,
								trainable=False)

	# 获取一个100个整数组成的序列. 每个整数代表一行词向量
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	x = Convolution1D(nb_filter = NB_FILTER,
					  filter_length=1,
					  activation='relu',
					  border_mode = 'valid')(embedded_sequences)
	x1 = Convolution1D(nb_filter=NB_FILTER,
					  filter_length=FILTER_LENGTH[0],
					  activation='relu',
					  border_mode='valid')(x)
	x2 = Convolution1D(nb_filter=NB_FILTER,
					   filter_length=FILTER_LENGTH[1],
					   activation='relu',
					   border_mode='valid')(x)
	x3 = Convolution1D(nb_filter=NB_FILTER,
					   filter_length=FILTER_LENGTH[2],
					   activation='relu',
					   border_mode='valid')(x)

	x = merge([x1, x2, x3], mode='concat', concat_axis=1)

	x = Bidirectional(LSTM(100))(x)
	x = Dropout(0.25)(x)
	x = Dense(100, activation='relu')(x)
	x = Dense(50, activation='relu')(x)
	preds = Dense(labels_num, activation='softmax')(x)
	model = Model(sequence_input, preds)

	return model