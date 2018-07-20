#!/bin/bash

# 获取文件名
file_name=$1

# 获取文件的列数
cols=`awk -F ',' '{print NF}' $file_name |head -1`

# 循环计算每个字段的去重值
for i in `seq 1 $cols`
do
	col_name=`sed -n "1, 1p" $file_name | awk -F ',' '{print $"'$i'"}'`
	uniq_cnt=`cat $file_name | awk -F ',' '{if (NR>1) { print $"'$i'"}}' |sort| uniq | wc -l|awk '{print $1}'`
	printf "%-15s %-30s %-20s\n" $i $col_name $uniq_cnt >>"../out_put/feature_ana.dat"
done