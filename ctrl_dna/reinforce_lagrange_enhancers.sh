
for seed in 0 1 8 42 617
do
    python reinforce_multi_lagrange.py --beta 0.01 --max_iter 100 --task hepg2 --oracle_type paired --grpo True --epoch 5 --wandb_log --lambda_lr 3e-4 --lambda_value 0.5 0.5 --tfbs_ratio 0.1  --seed $seed 
    python reinforce_multi_lagrange.py --beta 0.01 --max_iter 100 --task k562   --oracle_type paired --grpo True --epoch 5 --wandb_log --lambda_lr 3e-4 --lambda_value 0.5 0.5 --tfbs_ratio 0.1  --seed $seed 
    python reinforce_multi_lagrange.py --beta 0.01 --max_iter 100 --task sknsh  --oracle_type paired --grpo True --epoch 5 --wandb_log --lambda_lr 3e-4 --lambda_value 0.7 0.3 --tfbs_ratio 0.1  --seed $seed 
done
