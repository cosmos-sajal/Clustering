#/bin/bash

touch irisModified.txt
while read line           
do           
    echo $line | cut -d ',' -f1,2,3,4 >> irisModified.txt
done < $1