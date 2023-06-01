# author:Lenovo
# datetime:2023/3/28 8:37
# software: PyCharm
# project:pytorch项目


train_filepath='./Kcr_CV.fa'
test_filepath='./Kcr_IND.fa'

def read_data(filepath):

    with open(filepath,mode='r',encoding='utf-8') as f:
        index=1
        sequence=[]
        label=[]
        for line in f.readlines():

            if(index % 2 !=0):
                #如果是标签；
                str=line.strip('\n').split('|')
                # print(str)
                label.append((str[1]))
            else:
                #如果是氨基酸序列就直接加入：
                sequence.append(line.strip('\n'))
            index += 1
    # print(label)
    #
    # print(sequence)
    # print("len:",len(label))
    return sequence,label


train_sequence,train_label=read_data(train_filepath)
test_sequence,test_label=read_data(test_filepath)

def save_data(filepath,sequence,label):

    with open(filepath,mode='r+',encoding='utf-8') as f:

        for i in range(len(label)):
            #先读后写：
            f.read()
            contend=sequence[i]+','+label[i]
            f.write(contend)
            f.write('\n')
            f.flush()
        f.close()
        print("writer over")

save_data('./new_data/Kcr_cv.csv',train_sequence,train_label)
save_data('./new_data/Kcr_ind.csv',test_sequence,test_label)
