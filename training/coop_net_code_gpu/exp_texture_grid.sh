#!/bin/bash
show_accounts
squeue -u enijkamp
sbatch exp_texture_grid.sb
squeue -u enijkamp
