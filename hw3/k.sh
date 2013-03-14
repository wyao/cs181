for N in 2 3 4 5 6 7 8 9 10
do
    echo $N
    for M in 1 2 3 4 5
    do
        python clust.py $N 1000 -k
    done
done