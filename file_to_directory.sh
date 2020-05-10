source activate torch
for name in $(ls $1*json)
#for name in $(ls $1)
do
#name=$1$name
#if [[ -f $name && $name != *"json" && $name != *"bz2" ]]
#then
echo $name
#mv $name $name.json
python ensemble/split_file_to_directory.py $name $1
#python ensemble/split_file_to_directory.py $name.json $1
#fi
done
