#!/usr/bin/env bash

# This shell script both generates a sbatch script, and uses it to launch a local
# python script. It always overwrites the sbatch script that exists (note that
# the first redirection overwrites) and tries to ensure that it uses absolute
# file paths + pushd/popd to avoid shenanigans from the caller location. It also
# adds a tiny bit of logging, by appending the output and error to a record file,
# including the date/time, and neat dividers to separate the output from
# different invocations.

scriptname="submit-plot.sbatch"
projectname="tinb-plt"
projecthome=/home/i.benjamin/projects
runhome=$projecthome/tinb-mingwang/plotter
output=$runhome/output
timebar="\"==================Current Time:\$(date)==================\""
endbar="\"================================================================================\""

echo "#!/usr/bin/env bash" > $scriptname
echo "#SBATCH --partition=short" >> $scriptname
echo "#SBATCH --nodes=1" >> $scriptname
echo "#SBATCH --cpus-per-task=1" >> $scriptname
echo "#SBATCH --output=$output/output.txt" >> $scriptname
echo "#SBATCH --error=$output/error.txt" >> $scriptname
echo "#SBATCH --mem=3G" >> $scriptname
echo "pushd $runhome" >> $scriptname
echo "if [ -d \"$output\" ]; then" >> $scriptname
echo "    echo \"output directory exists\"" >> $scriptname
echo "else" >> $scriptname
echo "    echo \"no output directory\"" >> $scriptname
echo "    echo \"creating output directory\"" >> $scriptname
echo "    mkdir output" >> $scriptname
echo "fi" >> $scriptname
echo "echo \"loading modules...\"" >> $scriptname
echo "module load anaconda3/2024.06" >> $scriptname
echo "echo \"modules loaded.\"" >> $scriptname
echo "pushd $projecthome" >> $scriptname
echo "echo \"activating environment...\"" >> $scriptname
echo "source activate ./plotter" >> $scriptname
echo "echo \"environment activated.\"" >> $scriptname
echo "popd" >> $scriptname
echo "echo \"starting plot.py...\"" >> $scriptname
echo "python3 plot.py" >> $scriptname
echo "echo \"finished plotting\"" >> $scriptname
echo "popd" >> $scriptname
echo "echo $timebar >> $output/record_output.txt" >> $scriptname
echo "cat $output/output.txt >> $output/record_output.txt" >> $scriptname
echo "echo $endbar >> $output/record_output.txt" >> $scriptname
echo "echo $timebar >> $output/record_error.txt" >> $scriptname
echo "cat $output/error.txt >> $output/record_error.txt" >> $scriptname
echo "echo $endbar >> $output/record_error.txt" >> $scriptname


sbatch $scriptname
