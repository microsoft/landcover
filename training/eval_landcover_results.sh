#!/bin/bash

#echo "Accuracy and jaccard of all pixels"
tail -n 8 ${1} | head -n 4 | awk '{
    gsub("\\[", "")
    gsub("\\]", "")
    h[NR" "1] = $1
    h[NR" "2] = $2
    h[NR" "3] = $3
    h[NR" "4] = $4
    all_n += $1+$2+$3+$4
}
END{
    c=1
    j1 = h[c" "c] / (h[c" "1]+h[c" "2]+h[c" "3]+h[c" "4]+h[1" "c]+h[2" "c]+h[3" "c]+h[4" "c]-h[c" "c])
    c=2
    j2 = h[c" "c] / (h[c" "1]+h[c" "2]+h[c" "3]+h[c" "4]+h[1" "c]+h[2" "c]+h[3" "c]+h[4" "c]-h[c" "c])
    c=3
    j3 = h[c" "c] / (h[c" "1]+h[c" "2]+h[c" "3]+h[c" "4]+h[1" "c]+h[2" "c]+h[3" "c]+h[4" "c]-h[c" "c])
    c=4
    j4 = h[c" "c] / (h[c" "1]+h[c" "2]+h[c" "3]+h[c" "4]+h[1" "c]+h[2" "c]+h[3" "c]+h[4" "c]-h[c" "c])
    diag = h[1" "1] + h[2" "2] + h[3" "3] + h[4" "4]
    printf("%.6f,%.6f\n", diag / all_n, (j1+j2+j3+j4) / 4.0)
}'

#echo "Accuracy and jaccard of pixels with developed NLCD classes"
tail -n 4 ${1} | head -n 4 | awk '{
    gsub("\\[", "")
    gsub("\\]", "")
    h[NR" "1] = $1
    h[NR" "2] = $2
    h[NR" "3] = $3
    h[NR" "4] = $4
    all_n += $1+$2+$3+$4
}
END{
    c=1
    j1 = h[c" "c] / (h[c" "1]+h[c" "2]+h[c" "3]+h[c" "4]+h[1" "c]+h[2" "c]+h[3" "c]+h[4" "c]-h[c" "c])
    c=2
    j2 = h[c" "c] / (h[c" "1]+h[c" "2]+h[c" "3]+h[c" "4]+h[1" "c]+h[2" "c]+h[3" "c]+h[4" "c]-h[c" "c])
    c=3
    j3 = h[c" "c] / (h[c" "1]+h[c" "2]+h[c" "3]+h[c" "4]+h[1" "c]+h[2" "c]+h[3" "c]+h[4" "c]-h[c" "c])
    c=4
    j4 = h[c" "c] / (h[c" "1]+h[c" "2]+h[c" "3]+h[c" "4]+h[1" "c]+h[2" "c]+h[3" "c]+h[4" "c]-h[c" "c])
    diag = h[1" "1] + h[2" "2] + h[3" "3] + h[4" "4]
    printf("%.6f,%.6f\n", diag / all_n, (j1+j2+j3+j4) / 4.0)
}'

exit 0