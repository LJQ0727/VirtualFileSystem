
run_test: test
	./test > test.log
test: main.cu file_system.cu user_program.cu file_system.h
	nvcc --relocatable-device-code=true main.cu user_program.cu file_system.cu -o test
clean:
	rm test test.log snapshot.bin