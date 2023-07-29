# author:Lenovo
# datetime:2023/7/17 16:49
# software: PyCharm
# project:pytorch项目


import re

def deal_data(filename,save_filename):

    with open(filename,mode='r',encoding='utf-8') as f:

        with open(save_filename,mode='a+',encoding='utf-8') as f2:
            i=1
            for line in f.readlines():

                if i %2 == 0:
                    # print(line.strip())
                    line=line.strip()
                    new_line=re.sub('[^ACDEFGHIKLMNPQRSTVWY]','X',line)
                    # print(new_line)
                    f2.write(new_line+"\n")
                    i+=1
                else:
                    f2.write(line)
                    i+=1
        print("处理文件完成")
        f2.close()
        f.close()
if __name__ == '__main__':
    pos_filepath="../Datas/Kcr_positive_training.fasta"
    neg_filepath = "../Datas/Kcr_negative_training.fasta"


    save_pos_filepath="../Datas/Kcr_Pos_training.fasta"
    save_neg_filepath="../Datas/Kcr_Neg_training.fasta"

    deal_data(pos_filepath,save_pos_filepath)

    deal_data(neg_filepath, save_neg_filepath)
