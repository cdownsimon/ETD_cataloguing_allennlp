cmd="$@"
$cmd
ret=$?
while [ $ret -ne 0 ]
do 
    $cmd --recover
    ret=$?
done    
