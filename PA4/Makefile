INCLUDE=-I/usr/local/cuda/8.0/cuda/include 

SOURCE=dense_matvec.cu
EXECUTABLE=dense

$(EXECUTABLE): $(SOURCE)
	nvcc $(INCLUDE) $< -o $@ 

clean:
