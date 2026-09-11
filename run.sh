#!/usr/bin/env sh

# Nx_all=1024
# Ny_all=768

Nx_all=$((12 * 64))
Ny_all=$((13 * 64))

material="Ti-Nb"
name="grad-low"
gpu_num=1
velocity=$(awk "BEGIN {printf 0.0842 }")  # m/s, pulling velocity
gradient=$(awk "BEGIN {printf 5*0.001 }") # K/nm, temperature gradient

tag="${material}:${name}-${gpu_num}-${velocity}-${gradient}"

debug=false
total_time=$(awk "BEGIN {printf 120*1000 }") # ns
if_load=0

eps1=$(awk "BEGIN {printf 0.012 }") # anisotropy of the interfacial free energy
eps2=$(awk "BEGIN {printf 0 }")     # anisotropy of the interfacial free energy
epk1=$(awk "BEGIN {printf 0.1 }")   # anisotropy of the interfacial free energy
path_input="$(pwd)/${tag}"
run_time=8
num_pending_threshold=10
sleep_time="10m"
partition=rtx-batch
random_seed=$(awk "BEGIN {printf 0 }")

######################################################################## sleep if too many jobs are waiting

pending() {
	num_pending=$(squeue --me -h -t pending -p ${partition} -r | wc -l)
}
pending

# echo $num_pending
# echo $num_pending_threshold
# echo "got to the pending loop"
while test "$num_pending" -gt $num_pending_threshold; do
	echo "There are currently too many jobs pending!"
	printf 'Sleeping for %s ...\n' "$sleep_time"
	sleep $sleep_time
	pending
done

#######################################################################

source_name="fun.cu"
job_name="sim-${tag}"
sbatch_name="init/submit_${tag}.sh"
exec_name="${path_input}/exec_${tag}"
out_name="${path_input}/out_temp_${tag}.txt"
error_name="${path_input}/error_${tag}.txt"

# currently NO debug switch in there, you'll have to add it yourself.
# discovery has cuda/12.1 as its most recent version
# explorer has cuda/13.2.0 as its most recent version
# aicr has cuda/13.1.1 as its most recent version
echo "Making sbatch script..."
cat <<EOF >"${sbatch_name}"
#!/bin/env sh
#SBATCH --job-name="${job_name}"
#SBATCH --partition=${partition}
#SBATCH --gres=gpu:${gpu_num}
#SBATCH --mem=8Gb
#SBATCH --nodes=1
#SBATCH --output="${out_name}"
#SBATCH --error="${error_name}"
#SBATCH --time=0${run_time}:00:00

module load cuda/13.1.1

nvcc -arch=sm_80 \\
    --std=c++17 \\
    -Dtotal_time=${total_time} \\
    -Dif_load=${if_load} \\
    -Dpath_input=\"${path_input}\" \\
    -DVp=${velocity} \\
    -DGG=${gradient} \\
    -Deps1=${eps1} \\
    -Deps2=${eps2} \\
    -Depk1=${epk1} \\
    -Drandom_seed=${random_seed} \\
    -DNx_all=${Nx_all} \\
    -DNy_all=${Ny_all} \\
    -Drun_time=${run_time} \\
    -DMAX_GPU=${gpu_num} \\
     ${source_name} -o ${exec_name}
stdbuf -oL ${exec_name} 0 > ${out_name}

echo "\n########################################\n" >> ${path_input}/out.txt
cat ${out_name} >> ${path_input}/out.txt
echo "\n########################################\n" >> ${path_input}/out.txt
EOF
echo "${sbatch_name} has been written."
####################################################################### RELAUNCH SETUP
if test -d "${path_input}"; then
	echo "Resetting all data in ${path_input}..."
	rm -r "${path_input}"
	echo "Reset."
fi
echo "Making data directory..."
mkdir -p "${path_input}/data"
echo "Making directory for the initializing sbatch scripts..."
mkdir "${path_input}/init"
# mkdir "${path_input}/src"
echo "Directories set."
echo "Writing new source file and sbatch script for the simulation restarts..."
sed -E -f ./src.sed "${source_name}" >"${path_input}/${source_name}"
# turns off starting at step 0
sed -E -f ./init.sed "${sbatch_name}" >"${path_input}/${sbatch_name}"
# puts in new input values for restarting the same simulation instead of making a new simulation.
echo "Files written."
#######################################################################
echo "Launching initial and repeat slurm jobs:"
# notif=$(sbatch "${sbatch_name}")
# firstjobnum=$(echo "${notif}" | awk '/[0-9.]+/ { print $4 }')
# echo "$notif"
# prev_dir="$(pwd)"
# cd "${path_input}/" || exit 1
# sbatch --depend=afterany:"$firstjobnum" "${sbatch_name}"
# cd "$prev_dir" || exit 1
echo "Jobs launched."
