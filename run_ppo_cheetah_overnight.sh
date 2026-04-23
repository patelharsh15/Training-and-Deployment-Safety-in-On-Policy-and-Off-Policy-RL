#!/bin/bash
source venv/bin/activate

echo "Starting massive 12-seed overnight run for HalfCheetah PPO..."

for seed in {0..11}; do
    echo "========================================================="
    echo "Processing Seed $seed out of 11..."
    echo "========================================================="
    python src/train_mujoco.py --env HalfCheetah-v4 --algo ppo --seed $seed \
                               --lr 3e-4 --gamma 0.99 --batch-size 256 \
                               --total-timesteps 1000000 --tag best
    echo "Seed $seed complete!"
done

echo "OVERNIGHT SCRIPT FULLY COMPLETED!"
