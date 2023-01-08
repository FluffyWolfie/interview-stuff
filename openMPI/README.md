## To compile code
`mpicc asgn2.c -o asgn2.out -lm`

## To run the code
`mpirun -np 13 --oversubscribe asgn2.out 3 4 60`

`mpirun -np x*y+1 --oversubscribe asgn2.out x y iter`

command line -> x, y, iteration count

processes must be x*y+1, so ie 3x4+1 = 13

x and y = sensor network node grid row and column

log_file.txt refreshes for each run