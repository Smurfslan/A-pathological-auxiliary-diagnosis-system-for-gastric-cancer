echo "开始"


echo "1......"
python /mnt/ai2019/ljl/code/software_platform/infer/Test_slide_pre_list1.py --test_file /mnt/ai2020/ljl/data/gastric/paper_test/3.7_small/tumor_tif/ --tumor_flag 1  --csv_flie ./matrix/small/Test1_Semi_ts_0.3_DeepLabV3Plus_b3_classification_small.csv
echo "1......"
python /mnt/ai2019/ljl/code/software_platform/infer/Test_slide_pre_list1.py --test_file /mnt/ai2020/ljl/data/gastric/paper_test/3.7_small/normal_tif/ --tumor_flag 0  --csv_flie ./matrix/small/Test1_Semi_ts_0.3_DeepLabV3Plus_b3_classification_small.csv

echo "结束"
