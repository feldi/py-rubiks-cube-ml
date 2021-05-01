@echo off

# tests and experiments


# generate cubes to solve
python ./gen_cubes.py -e cube2x2simple -n 12 -d 80 -o output/gen-c22s-12-80-1.txt
python ./gen_cubes.py -e cube2x2simple -n 10 -d 5 -o output/gen-c22s-10-5-1.txt
python ./gen_cubes.py -e cube2x2simple -n 30 -d 12 -o output/gen-c22s-30-12-1.txt


# training
python ./train.py -i ./ini/cube2x2simple-zg-d200.ini -n run1


# debug training
python ./train_debug.py -e cube2x2simple -m output/cube2x2simple-zg-d200-run1/best_so_far.txt -o output/c22s-debug.csv 


# solve, output csv
python ./solver.py -e cube2x2simple -m output/cube2x2simple-zg-d200-run1/best_so_far.txt --output output/c22s-solve1.csv --max-steps 30000 --cuda 
python ./solver.py -e cube2x2simple -m output/cube2x2simple-zg-d200-run1/best_so_far.txt --perm R+,B+,R-,B- --max-steps 30000 --cuda 
python ./solver.py -e cube2x2simple -m output/cube2x2simple-zg-d100-run1/best_so_far.txt --input output/gen-c22s-30-12-1.txt --max-steps 30000 --cuda 
python ./solver.py -e cube2x2simple -m output/cube2x2simple-zg-d200-run190721-1/best_so_far.txt --input output/gen-c22s-30-12-1.txt --max-steps 30000 --cuda 


# solve, output plot png
python ./solver.py -e cube2x2simple -m output/cube2x2simple-zg-d200-run1/best_so_far.txt --max-steps 30000 --cuda --plot output/c22s-plot-1

