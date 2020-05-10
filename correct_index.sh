for file in $(ls $1)
do
if [[ ! -f "$2$file" ]] 
then
echo $1$file
echo $2$file
fi
done
