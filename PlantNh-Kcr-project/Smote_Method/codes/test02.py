# author:Lenovo
# datetime:2023/7/9 16:44
# software: PyCharm
# project:pytorch项目


AA_aaindex = 'ACDEFGHIKLMNPQRSTVWY'

word2id_dict = {'[UNK]': 0}
for i in range(len(AA_aaindex)):
    word2id_dict[AA_aaindex[i]] = i + 1

print(word2id_dict)

id_word_dict = {0: 'X'}

for i in range(len(AA_aaindex)):
    # print(AA_aaindex[i])
    id_word_dict[i+1]=AA_aaindex[i]
print(id_word_dict)