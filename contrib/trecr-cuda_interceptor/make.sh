rm -f ../../bin/kernel-loader.exe
make clean
make -j8
cp kernel-loader.exe ../../bin/