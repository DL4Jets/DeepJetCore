
#!/bin/zsh

while true
do
test=`nvidia-smi | grep  MiB | awk 'FNR == 1 {print $9}' | sed -e 's/MiB//g'`
if [ $test -le 200 ]
then
	eval "$@"
	break
else
	sleep 1
fi
done