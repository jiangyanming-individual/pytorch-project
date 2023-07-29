# author:Lenovo
# datetime:2023/7/9 8:40
# software: PyCharm
# project:pytorch项目


"""
使用SMOTE算法生成训练集的
平衡样本
少数样本过采样
"""

from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from numpy import where



def read_file(filepath):
	sequence_list=[]
	label_list=[]
	with open(file=filepath,mode='r',encoding='utf-8') as f:

		for line in f.readlines():
			line=line.strip().split(',')
			sequence_list.append(line[0])
			label_list.append(int(line[1]))
			# print(sequence_list)
			# print(label_list)
	f.close()

	return sequence_list,label_list



def generate_datasets(sequence_list,label_list,save_filepath):

	AA_aaindex = 'ACDEFGHIKLMNPQRSTVWY'
	#word :id
	word2id_dict = {'X': 0}
	for i in range(len(AA_aaindex)):
		word2id_dict[AA_aaindex[i]] = i + 1

	# print(word2id_dict)
	#id:word
	id_word_dict = {0: 'X'}
	for i in range(len(AA_aaindex)):
		# print(AA_aaindex[i])
		id_word_dict[i + 1] = AA_aaindex[i]

	# print(id_word_dict)

	# summarize class distribution
	X=[]
	y=[]
	for seq in sequence_list:
		# print(seq)
		one_seq=[]
		for index in seq:
			if index in word2id_dict:
				one_seq.append(word2id_dict.get(index))
			else:
				one_seq.append(word2id_dict.get('X'))
		# X.append()
		# print(seq_id)
		X.append(one_seq)

	y=label_list

	# print("X：",X)
	# print(len(X))
	# print("y:",y)

	counter1= Counter(y)
	print("counter1:",counter1)
	#
	over=SMOTE(k_neighbors=5,random_state=42)

	#不使用欠采样
	# under = RandomUnderSampler(sampling_strategy=0.5)
	# steps=[('over',over),('under',under)]
	# piplines=Pipeline(steps=steps)


	new_X,new_y=over.fit_resample(X,y)
	counter2=Counter(new_y)
	# print("new_X:",X)
	# print("new_y:",new_y)
	print("coumter2:",counter2)

	with open(save_filepath,mode='a+',encoding='utf-8') as f:

		for seq,label in zip(new_X,new_y):
			# print(seq)
			# print(label)
			one_seq=""
			for index in seq:
				one_seq=one_seq+id_word_dict.get(index)

			# print(one_seq)
			f.write(one_seq+','+str(label)+"\n")
			# break
	f.close()
	print("数据生成完毕！")


if __name__ == '__main__':
	filepath='../train_datasets/commonWheats_train.csv'
	save_filepath= '../generate_train_datasets/commonWheats_new_train.csv'
	sequence_list, label_list = read_file(filepath)
	generate_datasets(sequence_list, label_list, save_filepath)



	filepath = '../train_datasets/papaya_train.csv'
	save_filepath = '../generate_train_datasets/papaya_new_train.csv'
	sequence_list, label_list = read_file(filepath)
	generate_datasets(sequence_list, label_list, save_filepath)

	filepath = '../train_datasets/peanut_train.csv'
	save_filepath = '../generate_train_datasets/peanut_new_train.csv'
	sequence_list, label_list = read_file(filepath)
	generate_datasets(sequence_list, label_list, save_filepath)



	filepath = '../train_datasets/rice_trian.csv'
	save_filepath = '../generate_train_datasets/rice_new_trian.csv'
	sequence_list, label_list = read_file(filepath)
	generate_datasets(sequence_list, label_list, save_filepath)



	filepath = '../train_datasets/tabacum_train.csv'
	save_filepath = '../generate_train_datasets/tabacum_new_train.csv'
	sequence_list, label_list = read_file(filepath)
	generate_datasets(sequence_list, label_list, save_filepath)



	filepath = '../train_datasets/wheatSeeding_train.csv'
	save_filepath = '../generate_train_datasets/wheatSeeding_new_train.csv'
	sequence_list, label_list = read_file(filepath)
	generate_datasets(sequence_list, label_list, save_filepath)



	"""
	generate ind_test
	"""
	# filepath = '../ind_test_datasets/commonWheats_test.csv'
	# save_filepath = '../generate_ind_test_datasets/commonWheats_new_test.csv'
	# sequence_list, label_list = read_file(filepath)
	# generate_datasets(sequence_list, label_list, save_filepath)
	#
	#
	# filepath = '../ind_test_datasets/tabacum_test.csv'
	# save_filepath = '../generate_ind_test_datasets/tabacum_new_test.csv'
	# sequence_list, label_list = read_file(filepath)
	# generate_datasets(sequence_list, label_list, save_filepath)
	#
	#
	# filepath = '../ind_test_datasets/rice_test.csv'
	# save_filepath = '../generate_ind_test_datasets/rice_new_test.csv'
	# sequence_list, label_list = read_file(filepath)
	# generate_datasets(sequence_list, label_list, save_filepath)
	#
	# filepath = '../ind_test_datasets/papaya_test.csv'
	# save_filepath = '../generate_ind_test_datasets/papaya_new_test.csv'
	# sequence_list, label_list = read_file(filepath)
	# generate_datasets(sequence_list, label_list, save_filepath)
	#
	# filepath = '../ind_test_datasets/peanut_test.csv'
	# save_filepath = '../generate_ind_test_datasets/peanut_new_test.csv'
	# sequence_list, label_list = read_file(filepath)
	# generate_datasets(sequence_list, label_list, save_filepath)
	#
	# filepath = '../ind_test_datasets/wheatSeeding_test.csv'
	# save_filepath = '../generate_ind_test_datasets/wheatSeeding_new_test.csv'
	# sequence_list, label_list = read_file(filepath)
	# generate_datasets(sequence_list, label_list, save_filepath)
