INCLUDE=-I/usr/local/cuda/8.0/cuda/include 

SOURCE=prac_cuda_matmul.cu
EXECUTABLE=matmul

$(EXECUTABLE): $(SOURCE)
	nvcc  -g $(INCLUDE) $< -o $@

clean:
