cat gamd-output.dat | awk '{print ($3+$5)*1000.0/8.314/300.0, $1, ($3+$5)/4.184}' > weights.dat
