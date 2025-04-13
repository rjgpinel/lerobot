#!/bin/bash

run_act_chicken() {
    rm -rf ~/.cache/huggingface/lerobot/SanjarNormuradov/eval_act_chickenToPlate
    python lerobot/scripts/control_robot.py \
      --robot.type=so100 \
      --control.type=record \
      --control.fps=30 \
      --control.single_task="Pick up the chicken drumstick and place on the plate. No any other objects in the scene." \
      --control.repo_id=SanjarNormuradov/eval_act_chickenToPlate \
      --control.tags='["tutorial"]' \
      --control.warmup_time_s=5 \
      --control.episode_time_s=60 \
      --control.reset_time_s=30 \
      --control.num_episodes=10 \
      --control.push_to_hub=false \
      --control.policy.path=/home/sanjar.normuradov/.cache/huggingface/lerobot/rjgpinel/chicken_to_plate_act/checkpoints/100000/pretrained_model
} # 040000, 100000, last

run_dp_chicken() {
    rm -rf ~/.cache/huggingface/lerobot/SanjarNormuradov/eval_dp_chickenToPlate
    python lerobot/scripts/control_robot.py \
        --robot.type=so100 \
        --control.type=record \
        --control.fps=30 \
        --control.single_task="Pick up the chicken drumstick and place on the plate. No any other objects in the scene." \
        --control.repo_id=SanjarNormuradov/eval_dp_chickenToPlate \
        --control.tags='["tutorial"]' \
        --control.warmup_time_s=5 \
        --control.episode_time_s=60 \
        --control.reset_time_s=30 \
        --control.num_episodes=10 \
        --control.push_to_hub=false \
        --control.policy.path=/home/sanjar.normuradov//.cache/huggingface/lerobot/dpSo100ChickenToPlate/checkpoints/010400/pretrained_model
}

run_act_cube() {
    rm -rf ~/.cache/huggingface/lerobot/SanjarNormuradov/eval_act_cubeToPlate
    python lerobot/scripts/control_robot.py \
        --robot.type=so100 \
        --control.type=record \
        --control.fps=30 \
        --control.single_task="Pick up the red cube and place it in the white bowl." \
        --control.repo_id=SanjarNormuradov/eval_act_cubeToPlate \
        --control.tags='["tutorial"]' \
        --control.warmup_time_s=5 \
        --control.episode_time_s=60 \
        --control.reset_time_s=30 \
        --control.num_episodes=10 \
        --control.push_to_hub=false \
        --control.policy.path=/home/sanjar.normuradov/.cache/huggingface/lerobot/rjgpinel/cubePickNight/080000/pretrained_model
}

set_static_cam_controls() {
  echo "Setting static_right camera controls..."
  v4l2-ctl --device=/dev/video2 --set-ctrl brightness=-32,contrast=32,saturation=64,gamma=100

  echo "Setting static_left camera controls..."
  v4l2-ctl --device=/dev/video4 --set-ctrl brightness=-32,contrast=32,saturation=64,gamma=100
}

case "$1" in
  run_act_chicken)
    run_act_chicken
    ;;
  run_dp_chicken)
    run_dp_chicken
    ;;
  run_act_cube)
    run_act_cube
    ;;
  set_static_cam_controls)
    set_static_cam_controls
    ;;
  *)
    echo "Unknown command: $1"
    echo "Available commands: run_act_chicken, run_dp_chicken, run_act_cube"
    exit 1
    ;;
esac
