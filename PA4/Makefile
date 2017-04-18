INCLUDE=-I/usr/local/cuda/8.0/cuda/include 

SOURCE=spmv.cu
EXECUTABLE=sparse

$(EXECUTABLE): $(SOURCE)
	nvcc  -g -Wall $(INCLUDE) $< -o $@

clean:
