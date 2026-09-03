#!/usr/bin/env sh


# Nx_all=1024
# Ny_all=768

Nx_all=$((12 * 64))
Ny_all=$((13 * 64))

material="Ti-Nb"
name="flat-small"
gpu_num=1
velocity=$(awk "BEGIN {printf 0.240 }") # m/s, pulling velocity
gradient=$(awk "BEGIN {printf 10*0.001 }") # K/nm, temperature gradient

tag="${material}:${name}-${gpu_num}-${velocity}-${gradient}"

debug=false
total_time=$(awk "BEGIN {printf 120*1000 }") # ns
if_load=0

eps1=$(awk "BEGIN {printf 0.012 }") # anisotropy of the interfacial free energy
eps2=$(awk "BEGIN {printf 0 }") # anisotropy of the interfacial free energy
epk1=$(awk "BEGIN {printf 0.1 }") # anisotropy of the interfacial free energy
path_input="$(pwd)/${tag}"
run_time=8
num_pending_threshold=10
sleep_time="10m"
partition=multigpu,gpu
random_seed=$(awk "BEGIN {printf 0 }")

######################################################################## sleep if too many jobs are waiting

pending () {
    num_pending=$(squeue --me -h -t pending -p ${partition} -r | wc -l)
}
pending
# echo $num_pending
# echo $num_pending_threshold
# echo "got to the pending loop"
while (( num_pending >= num_pending_threshold )); do
    echo "sleeping ${sleep_time}\n"
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
cat << EOF > ${sbatch_name}
#!/bin/env sh
#SBATCH --job-name="${job_name}"
#SBATCH --partition=${partition}
#SBATCH --gres=gpu:a100:${gpu_num}
#SBATCH --mem=8Gb
#SBATCH --nodes=1
#SBATCH --output="${out_name}"
#SBATCH --error="${error_name}"
#SBATCH --time=0${run_time}:00:00

module load cuda/13.2.0

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

####################################################################### RELAUNCH SETUP
mkdir "${path_input}"
mkdir "${path_input}/data"
mkdir "${path_input}/init"
# mkdir "${path_input}/src"

sed '25c\
#define    if_start_from_step0      0' "${source_name}" > "${path_input}/${source_name}" # turns off starting at step 0
sed '16c\
    -Dif_load=1 \\
17c\
    -Dpath_input=\\\"./.\\\" \\
10i#SBATCH --array=1-3%1' "${sbatch_name}" > "${path_input}/${sbatch_name}"
#######################################################################

notif=$(sbatch "${sbatch_name}")
firstjobnum=$(echo "${notif}" | awk '/[0-9.]+/ { print $4 }')
echo $notif
pushd "${path_input}/"
sbatch --depend=afterany:$firstjobnum "${sbatch_name}"
popd
