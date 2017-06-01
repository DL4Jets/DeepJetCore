
#!/bin/zsh

fullinput="$@"

#the nvidia-smi command is slow, only call it once
printout=$(nvidia-smi  | head -n 31 | tail -n 25 2>&1)

cutoff=23
for (( i=0; i< 9; i++))
do
test=`echo "${printout}" | tail -n ${cutoff} | head -n 1 | awk 'FNR == 1 {print $9}' | sed -e 's/MiB//g'`
cutoff=$((cutoff-3))

if [ $test -le 1 ]
	then
		echo "assigning to GPU ${i}"
		export CUDA_VISIBLE_DEVICES=$i
		break
    fi
   CUDA_VISIBLE_DEVICES=$i
done
if [ $CUDA_VISIBLE_DEVICES -ge 8 ]
	then
		echo "no free GPU available right now"
		exit
	fi
	
eval "$fullinput"