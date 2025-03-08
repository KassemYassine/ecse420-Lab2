all: sequential parallel parallel_512

sequential:
	nvcc grid_4_4_seq.cu -o grid_4_4_seq.exe

parallel:
	nvcc grid_4_4.cu -o grid_4_4.exe

parallel_512:
	nvcc grid_512_512.cu -o grid_512_512.exe