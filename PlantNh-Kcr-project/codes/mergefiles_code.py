# author:Lenovo
# datetime:2023/7/9 19:15
# software: PyCharm
# project:pytorch项目

"""
merged train and test datasets
"""

def merge_files(merged_filepath,file_list):

    merged_file = open(merged_filepath, "w")

    for file_name in file_list:
        # 打开当前要合并的文本文件
        current_file = open(file_name, "r")
        merged_file.write(current_file.read())

        # 关闭当前文件
        current_file.close()

    merged_file.close()
    print("合并文件完成！")
if __name__ == '__main__':

    #merge train_datasets
    merged_train_filepath= "../csv/total_train_ind_test/train.csv"

    train_file1 = '../csv/split_train_test/commonWheats_train.csv'
    train_file2 = '../csv/split_train_test/papaya_train.csv'
    train_file3 = '../csv/split_train_test/peanut_train.csv'
    train_file4 = '../csv/split_train_test/rice_train.csv'
    train_file5 = '../csv/split_train_test/tabacum_train.csv'
    train_file6 = '../csv/split_train_test/wheatSeeding_train.csv'

    # with open(train_file1,mode='r') as f:
    #     for line in f.readlines():
    #         print(line)
    #         break

    file_list=[train_file1,train_file2,train_file3,train_file4,train_file5,train_file6]
    merge_files(merged_train_filepath,file_list)


    # merge test_datasets
    merged_test_filepath = "../csv/total_train_ind_test/ind_test.csv"

    test_file1='../csv/split_train_test/commonWheats_test.csv'
    test_file2 = '../csv/split_train_test/papaya_test.csv'
    test_file3 = '../csv/split_train_test/peanut_test.csv'
    test_file4 = '../csv/split_train_test/rice_test.csv'
    test_file5 = '../csv/split_train_test/tabacum_test.csv'
    test_file6 = '../csv/split_train_test/wheatSeeding_test.csv'

    file_list=[test_file1,test_file2,test_file3,test_file4,test_file5,test_file6]
    merge_files(merged_test_filepath,file_list)