#!/usr/bin/env bash

# This shell script both generates a sbatch script, and uses it to launch a
# local python script. It always overwrites the sbatch script that exists (note
# that the redirection overwrites) and tries to ensure that it uses absolute
# file paths + pushd/popd to avoid shenanigans from the caller location. It also
# adds a tiny bit of logging, by appending the output and error to a record
# file, including the date/time, and neat dividers to separate the output from
# different invocations.

scriptname="submit-plot.sbatch"
output="$(pwd)/output"
timebar="\"==================Current Time:\$(date)==================\""
endbar="\"================================================================================\""

cat <<EOF >$scriptname
#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=${output}/output.txt
#SBATCH --error=${output}/error.txt
#SBATCH --mem=10G
if [ -d "${output}" ]; then
    echo "output directory exists"
else
    echo "no output directory"
    echo "creating output directory"
    mkdir output
fi
echo "loading modules..."
module load miniforge3/25.3.0-3
echo "modules loaded."
echo "activating environment..."
source activate plotting
echo "environment activated."
echo "starting plot.py..."
python3 plot.py
echo "finished plotting"
echo ${timebar} >> ${output}/record_output.txt
cat ${output}/output.txt >> ${output}/record_output.txt
echo ${endbar} >> ${output}/record_output.txt
echo ${timebar} >> ${output}/record_error.txt
cat ${output}/error.txt >> ${output}/record_error.txt
echo ${endbar} >> ${output}/record_error.txt
EOF

sbatch $scriptname
