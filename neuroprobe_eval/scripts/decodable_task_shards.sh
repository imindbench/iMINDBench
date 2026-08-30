#!/usr/bin/env bash

# Source this file, then export TASKS="${MACHINE_A_TASKS}" or
# TASKS="${MACHINE_B_TASKS}" before launching hold-in and multisource scripts.
MACHINE_A_TASKS="onset volume global_flow local_flow word_gap delta_volume frame_brightness word_head_pos"
MACHINE_B_TASKS="speech face_num pitch word_index word_part_speech word_length gpt2_surprisal"
