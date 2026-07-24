#!/usr/bin/env bash

# Random ranges
X_MIN=0.2
X_MAX=0.2
Y_MIN=-0.1
Y_MAX=-0.1
YAW_MIN=1.0
YAW_MAX=1.0

nic_yaw_min=-0.17
nic_yaw_max=0.17
nic_translation_min=-0.02
nic_translation_max=0.02

sfp_translaion_max_0=-0.09
sfp_translation_min_0=-0.03

sfp_translation_max_1=0.09
sfp_translation_min_1=-0.09

sfp_translation_max_2=0.09
sfp_translation_min_2=0.0

# Generate random values
task_board_x=$(awk -v min="$X_MIN" -v max="$X_MAX" 'BEGIN { srand(); print min + rand() * (max - min) }')
task_board_y=$(awk -v min="$Y_MIN" -v max="$Y_MAX" 'BEGIN { srand(); print min + rand() * (max - min) }')
task_board_yaw=$(awk -v min="$YAW_MIN" -v max="$YAW_MAX" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_yaw_0=$(awk -v min="$nic_yaw_min" -v max="$nic_yaw_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_yaw_1=$(awk -v min="$nic_yaw_min" -v max="$nic_yaw_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_yaw_2=$(awk -v min="$nic_yaw_min" -v max="$nic_yaw_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_yaw_3=$(awk -v min="$nic_yaw_min" -v max="$nic_yaw_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_yaw_4=$(awk -v min="$nic_yaw_min" -v max="$nic_yaw_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_translation_0=$(awk -v min="$nic_translation_min" -v max="$nic_translation_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_translation_1=$(awk -v min="$nic_translation_min" -v max="$nic_translation_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_translation_2=$(awk -v min="$nic_translation_min" -v max="$nic_translation_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_translation_3=$(awk -v min="$nic_translation_min" -v max="$nic_translation_max" 'BEGIN { srand(); print min + rand() * (max - min) }')
nic_translation_4=$(awk -v min="$nic_translation_min" -v max="$nic_translation_max" 'BEGIN { srand(); print min + rand() * (max - min) }')

sfp_rail_0_translation=$(awk -v min="$sfp_translation_min_0" -v max="$sfp_translaion_max_0  " 'BEGIN { srand(); print min + rand() * (max - min) }' )
sfp_rail_1_translation=$(awk -v min="$sfp_translation_min_1" -v max="$sfp_translaion_max_1" 'BEGIN { srand(); print min + rand() * (max - min) }' )
sfp_rail_2_translation=$(awk -v min="$sfp_translation_min_2" -v max="$sfp_translaion_max_2" 'BEGIN { srand(); print min + rand() * (max - min) }' )
# /entrypoint.sh \
#   spawn_cable:=true \

  # sfp_mount_rail_3_present:=true \
  # sfp_mount_rail_3_translation:=0.07 \

/entrypoint.sh \
  spawn_task_board:=true \
  task_board_x:=${task_board_x} \
  task_board_y:=${task_board_y} \
  task_board_roll:=0.0 \
  task_board_pitch:=0.0 \
  sfp_mount_rail_0_present:=true \
  sfp_mount_rail_0_translation:=${sfp_rail_0_translation} \
  sfp_mount_rail_2_present:=true \
  sfp_mount_rail_2_translation:=${sfp_rail_2_translation} \
  sfp_mount_rail_1_present:=true \
  sfp_mount_rail_1_translation:=${sfp_rail_1_translation} \
  sfp_mount_rail_4_present:=true \
  sfp_mount_rail_4_translation:=${sfp_rail_2_translation} \
  sc_mount_rail_0_present:=true \
  sc_mount_rail_0_translation:=-0.07 \
  sc_mount_rail_1_present:=true \
  sc_mount_rail_1_translation:=-0.04 \
  sc_mount_rail_2_present:=true \
  sc_mount_rail_2_translation:=0.0 \
  sc_mount_rail_3_present:=true \
  sc_mount_rail_3_translation:=0.07 \
  sc_mount_rail_4_present:=true \
  sc_mount_rail_4_translation:=0.04 \
  sc_port_0_present:=true \
  sc_port_0_translation:=-0.05 \
  sc_port_1_present:=true \
  sc_port_1_translation:=0.0 \
  sc_port_2_present:=true \
  sc_port_2_translation:=0.05 \
  sc_port_3_present:=true \
  sc_port_3_translation:=-0.03 \
  sc_port_4_present:=true \
  sc_port_4_translation:=0.03 \
  nic_card_mount_0_present:=true \
  nic_card_mount_0_translation:=${nic_translation_0} \
  nic_card_mount_0_yaw:=${nic_yaw_0} \
  nic_card_mount_1_present:=true \
  nic_card_mount_1_translation:=${nic_translation_1} \
  nic_card_mount_1_yaw:=${nic_yaw_1} \
  nic_card_mount_2_present:=true \
  nic_card_mount_2_translation:=${nic_translation_2} \
  nic_card_mount_2_yaw:=${nic_yaw_2} \
  nic_card_mount_3_present:=true \
  nic_card_mount_3_translation:=${nic_translation_3} \
  nic_card_mount_3_yaw:=${nic_yaw_3} \
  spawn_cable:=true \
  cable_name:=cable_0 \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_4_present:=true \
  nic_card_mount_4_translation:=${nic_translation_4} \
  nic_card_mount_4_yaw:=${nic_yaw_4} \
  ground_truth:=true \
  start_aic_engine:=false \
  task_board_yaw:=${task_board_yaw}
