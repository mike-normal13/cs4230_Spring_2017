INCLUDE=-I/usr/local/cuda/8.0/cuda/include 

SOURCE=spmv-par.c
EXECUTABLE=sparse

$(EXECUTABLE): $(SOURCE)
	nvcc $(INCLUDE) $< -o $@ 

clean:
