"""
created on:2018/8/20
author:DilicelSten
target:preprocess sougou data and turn to csv file
finished on:2018/8/20
"""
import csv
import numpy as np
import jieba
"""
data format:
(1)train
id, age, gender, education, query
age:[0：未知年龄; 1：0-18岁; 2：19-23岁; 3：24-30岁; 4：31-40岁; 5：41-50岁; 6： 51-999岁]
gender:[0：未知1：男性2：女性]
education:[0：未知学历; 1：博士; 2：硕士; 3：大学生; 4：高中; 5：初中; 6：小学]
"""

first_train_path = "/media/iiip/文档/user_profiling/sougou/processed data/user_tag_query_train"
first_csv_path = '/media/iiip/文档/user_profiling/sougou/processed data/first_train.csv'

second_train_path = "/media/iiip/文档/user_profiling/sougou/processed data/user_tag_query_2w_train"
second_csv_path = '/media/iiip/文档/user_profiling/sougou/processed data/second_train.csv'


def train2csv():
    """
    turn train dataset to csv file
    :return:
    """
    header = ["id", "age", "gender", "education", "query_content"]
    file = open(second_csv_path.replace(".csv", "_hanlp_cut.csv"), 'w')
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    data = []
    shuffle_data = []
    with open(second_train_path) as f:
        contents = f.readlines()
        for i in range(len(contents)):
            line = []
            query = ""
            content = contents[i].split('\t')
            for j in range(4):
                line.append(content[j])
            for s in content[4:]:
                query += s
            line.append(','.join(jieba.cut(query)))
            dic = dict(map(lambda x, y: [x, y], header, line))
            data.append(dic)
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    print(shuffle_indices)
    for each in shuffle_indices:
        shuffle_data.append(data[each])
    writer.writerows(shuffle_data)


if __name__ == '__main__':
    train2csv()