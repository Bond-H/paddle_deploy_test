from LAC import LAC
from time import time
lac = LAC('infer_model')

data = open('infer.tsv','r',encoding='utf8').readlines()
time3 = time()
for i in range(10):
    result2 = lac.lac_seg(data[i*100:(i+1)*100])
time4 = time()
print(time4-time3)

# 对于单个样本输入
time1 = time()
for text in data:
    result = lac.lac_seg([text])
time2 = time()
print(time2-time1)


