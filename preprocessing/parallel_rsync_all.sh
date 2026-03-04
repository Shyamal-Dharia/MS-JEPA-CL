#!/bin/bash
# Complete parallel rsync: all 1627 subjects (sub-0001 to sub-2062)
# One Duo auth, 5 parallel streams

SRC="shyamal@fir.alliancecan.ca:/scratch/shyamal/final_files/fmriprep_1600/"
DST="/data/s.dharia-ra/multi-site-problem/preprocessed_data_fir/fmriprep_1600/"
MUX="/tmp/sshmux_fir"
SSH="ssh -S $MUX -T -o Compression=no -x"

pkill -f "rsync.*fmriprep_1600" 2>/dev/null
rm -f "$MUX"
sleep 2

echo "=== Authenticate ONCE (password + Duo) ==="
ssh -fNM -S "$MUX" -T -o Compression=no -o ControlPersist=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 shyamal@fir.alliancecan.ca
if [ $? -ne 0 ]; then echo "SSH failed!"; exit 1; fi
echo "=== SSH ready ==="
mkdir -p "$DST"

# Stream 1: sub-0001 to sub-0400
rsync -aHX --partial -e "$SSH" --include='sub-0[0-3]*/' --include='sub-0400/' --include='sub-0[0-3]*/**' --include='sub-0400/**' --exclude='*' "$SRC" "$DST" > ~/rsync_s1.log 2>&1 &
echo "Stream 1 (sub-0001..0400) PID=$!"

# Stream 2: sub-0401 to sub-0800
rsync -aHX --partial -e "$SSH" --include='sub-0[4-7]*/' --include='sub-0800/' --include='sub-0[4-7]*/**' --include='sub-0800/**' --exclude='*' "$SRC" "$DST" > ~/rsync_s2.log 2>&1 &
echo "Stream 2 (sub-0401..0800) PID=$!"

# Stream 3: sub-0801 to sub-1200
rsync -aHX --partial -e "$SSH" --include='sub-0[8-9]*/' --include='sub-1[0-1]*/' --include='sub-1200/' --include='sub-0[8-9]*/**' --include='sub-1[0-1]*/**' --include='sub-1200/**' --exclude='*' "$SRC" "$DST" > ~/rsync_s3.log 2>&1 &
echo "Stream 3 (sub-0801..1200) PID=$!"

# Stream 4: sub-1201 to sub-1600
rsync -aHX --partial -e "$SSH" --include='sub-1[2-5]*/' --include='sub-1600/' --include='sub-1[2-5]*/**' --include='sub-1600/**' --exclude='*' "$SRC" "$DST" > ~/rsync_s4.log 2>&1 &
echo "Stream 4 (sub-1201..1600) PID=$!"

# Stream 5: sub-1601 to sub-2062
rsync -aHX --partial -e "$SSH" --include='sub-1[6-9]*/' --include='sub-2*/' --include='sub-1[6-9]*/**' --include='sub-2*/**' --exclude='*' "$SRC" "$DST" > ~/rsync_s5.log 2>&1 &
echo "Stream 5 (sub-1601..2062) PID=$!"

echo ""
echo "=== 5 streams running. Monitor: watch -n 60 'du -sh $DST' ==="
echo "Waiting for all to finish..."
wait
echo "=== ALL DONE ==="
ssh -S "$MUX" -O exit shyamal@fir.alliancecan.ca 2>/dev/null
