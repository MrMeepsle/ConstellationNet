#!/bin/bash
echo "Train GPU index:" $1 "Experiment:" $2 "Trial:" $3 "Test GPU index:" $4


python train_classifier_sideout_classifier_ybranch.py --gpu $1 --config ./configs/current_configs/featcluster_minibatch_res12_sideout_classifier_ybranch_self_att/$2.yaml --tag=$3

python test_few_shot_ybranch.py --shot 1 --gpu $4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-$2_$3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='1,1'

python test_few_shot_ybranch.py --shot 1 --gpu $4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-$2_$3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='2,2'

python test_few_shot_ybranch.py --shot 1 --gpu $4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-$2_$3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'

python test_few_shot_ybranch.py --shot 1 --gpu $4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-$2_$3/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'


python test_few_shot_ybranch.py --shot 5 --gpu $4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-$2_$3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='1,1'

python test_few_shot_ybranch.py --shot 5 --gpu $4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-$2_$3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool' --branch_list='2,2'

python test_few_shot_ybranch.py --shot 5 --gpu $4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-$2_$3/max-f-va.pth --method='cos,cos' --dist_func_list='none,none' --logits_coeff_list='20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,final' --branch_list='1,2'

python test_few_shot_ybranch.py --shot 5 --gpu $4 --config ./configs/current_configs/test_few_shot_mini_general.yaml --load=./save/res12featcluster-minibatch-sideout-classifier-ybranch-$2_$3/max-f-va.pth --method='cos,cos,cos,cos' --dist_func_list='none,none,none,none' --logits_coeff_list='20,20,20,20' --disable_out_feat_flatten --sideout --feat_source_list='final,before_avgpool,final,before_avgpool' --branch_list='1,1,2,2'