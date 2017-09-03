#!/bin/bash
show_accounts
squeue -u enijkamp
sbatch exp_texture.sb
squeue -u enijkamp
