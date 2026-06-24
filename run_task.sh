#!/usr/bin/env bash

# Random ranges
X_MIN=0.15
X_MAX=0.25
Y_MIN=-0.2
Y_MAX=-0.3
YAW_MIN=2.0
YAW_MAX=3.14

nic_yaw_min=-0.17
nic_yaw_max=0.17

nic_translation_min=-0.02
nic_translation_max=0.02

sc_port_translation_min=-0.06
sc_port_translation_max=0.05

# Generate random values
task_board_x=$(awk -v min="$X_MIN" -v max="$X_MAX" 'BEGIN { srand(); print min + rand() * (max - min) }')
task_board_y=$(awk -v min="$Y_MIN" -v max="$Y_MAX" 'BEGIN { srand(); print min + rand() * (max - min) }')
task_board_yaw=$(awk -v min="$YAW_MIN" -v max="$YAW_MAX" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_yaw=$(awk -v min="$nic_yaw_min" -v max="$nic_yaw_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_translation=$(awk -v min="$nic_translation_min" -v max="$nic_translation_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
sc_port_translation=$(awk -v min="$sc_port_translation_min" -v max="$sc_port_translation_max" 'BEGIN { srand(); print min + rand() * (max - min) }')

# /entrypoint.sh \
#   spawn_cable:=true \


/entrypoint.sh \
  spawn_task_board:=true \
  task_board_x:=${task_board_x} \
  task_board_y:=${task_board_y} \
  task_board_roll:=0.0 \
  task_board_pitch:=0.0 \
  sfp_mount_rail_0_present:=true \
  sfp_mount_rail_0_translation:=-0.00 \
  sc_mount_rail_0_present:=true \
  sc_mount_rail_0_translation:=0.0 \
  sc_port_1_present:=true \
  sc_port_1_translation:=${sc_port_translation} \
  sc_port_0_present:=true \
  sc_port_0_translation:=0.02 \
  spawn_cable:=true \
  attach_cable_to_gripper:=true \
  ground_truth:=true \
  start_aic_engine:=false \
  task_board_yaw:=${task_board_yaw}\
  cable_type:=sfp_sc_cable \
  nic_card_mount_4_present:=true \
  nic_card_mount_4_translation:=${nic_translation} \
  nic_card_mount_4_yaw:=${nic_yaw} \