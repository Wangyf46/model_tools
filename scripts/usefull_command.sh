#!/usr/bin/env bash

# 查找所有符合条件的文件，并复制到制定目录
find ./ReviewTsr/ -name "*.xml" | xargs -i sudo cp -v {} Annotations_Tsr/

# 根据requirement.txt安装依赖包
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements_gpu.txt

# 根据文件列表删除相关文件
while read line; do sudo rm $line.png; done < set2_diff_set1.txt

# 随即打乱文件行
awk 'BEGIN{srand()}{b[rand()NR]=$0}END{for(x in b)print b[x]}' test.txt | sudo tee shuffle.txt

# 生成列表文件
find JPEGIMages/ -name *.png | xargs -i basename {} .png | sudo tee  xx.txt

# 求两个文件列表的交集
comm -12 <(sort a.txt|uniq ) <(sort b.txt|uniq )

# 求两个文件列表的差集 a-b
comm -23 <(sort a.txt|uniq ) <(sort b.txt|uniq )

# 求两个文件列表的差集 b-a
comm -13 <(sort a.txt|uniq ) <(sort b.txt|uniq )

cat file | while read line; do sudo mv Annotations_Back/$line.xml Extral_Annotations; done