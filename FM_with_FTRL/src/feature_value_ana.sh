#!/bin/bash

# 获取文件名
file_name=$1

# 获取文件的列数
cols=`awk -F ',' '{print NF}' $file_name |head -1`

# 设置表头
out_put_path="../out_put/feature_ana.dat"
if [ -f "$out_put_path" ]; then
	rm $out_put_path
fi

echo -e "col_num\tvalue_num\tcol_name">>"../out_put/feature_ana.dat"

# 循环计算每个字段的去重值
for i in `seq 1 $cols`
do
	col_name=`sed -n "1, 1p" $file_name | awk -F ',' '{print $"'$i'"}'`
	uniq_cnt=`cat $file_name | awk -F ',' '{if (NR>1) { print $"'$i'"}}' |sort|uniq|wc -l|awk '{print $1}'`
	echo -e "$i\t$uniq_cnt\t$col_name">>"../out_put/feature_ana.dat"
done

# 调用.py脚本
python -u feature_value_ana.py


# 切分数据集
# awk 'BEGIN{FS=",";OFS=","}{$1=$2=$3="";print $0}' train.csv >feature.dat
# cut -d "," -f 4- feature.dat >feature_ryan.dat
# awk 'BEGIN{FS=",";OFS=","}{print $2}' train.csv >label_ryan_dat


