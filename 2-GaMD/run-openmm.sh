python gamd_eq-rf.py -o gamd_eq.out -p Cs-18c6_cl_wat15.prmtop -c md_npt3.rst -r gamd_eq.rst -x gamd_eq.netcdf -l 43-43
python gamd_prod-rf.py -o gamd_prod.out -p Cs-18c6_cl_wat15.prmtop -c gamd_eq.rst -r gamd_prod.rst -x gamd_prod.netcdf -l 43-43 -g gamd-para.txt
