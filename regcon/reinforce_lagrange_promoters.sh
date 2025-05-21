

python reinforce_multi_lagrange.py --beta 0.01 --max_iter 100 --task JURKAT --oracle_type paired --grpo True --epoch 5 --wandb_log --lambda_lr 3e-4 --lambda_value 0.1 0.9 --tfbs_ratio 0.1  
python reinforce_multi_lagrange.py --beta 0.01 --max_iter 100 --task K562  --oracle_type paired --grpo True --epoch 5 --wandb_log --lambda_lr 3e-3 --lambda_value 0.2 0.9 --tfbs_ratio 0.1   
python reinforce_multi_lagrange.py --beta 0.01 --max_iter 100 --task THP1  --oracle_type paired --grpo True --epoch 5 --wandb_log --lambda_lr 3e-3 --lambda_value 0.5 0.5 --tfbs_ratio 0.1  