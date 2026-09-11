/^[[:space:]]+-Dif_load=/s/.*/    -Dif_load=1 \\/
/^[[:space:]]+-Dpath_input=/s/.*/    -Dpath_input=".\/." \\/
10i\
#SBATCH --array=1-3%1
