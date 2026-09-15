#!/usr/bin/env bash
# Full study, detached. Every rung gets the same 32 restarts and the same
# 30-minute exact-solve attempt, 750x750 included. Rungs checkpoint after each
# instance, so a crash resumes instead of restarting.
set -u
cd /home/amaro/Pesquisa-Operacional-LP-GRASP
echo $$ > output/study.pid

echo "### STAGE SET 1: the library's official sizes (250, 500, 750) + California"
python -u -m scripts.run_full_study --sizes 250 500 750 \
    --california-restarts 10 --out-dir output

echo "### STAGE SET 2: provability frontier (100, 150, 200)"
python -u -m scripts.run_full_study --sizes 100 150 200 \
    --skip-california --out-dir output

echo "### STAGE SET 3: compose the complete six-rung ladder from every sidecar"
python -u -m scripts.run_full_study --sizes 100 150 200 250 500 750 \
    --from-stages --out-dir output

echo "### ALL STAGES COMPLETE"
rm -f output/study.pid
