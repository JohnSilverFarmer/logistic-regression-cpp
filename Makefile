
CXX = g++
CXXFLAGS = -I /usr/local/include/eigen3
BUILD_DIR = build

default: clean dir main

main: 
	$(CXX) $(CXXFLAGS) main.cpp -o build/main 

dir:
	mkdir -p ${BUILD_DIR}

clean:
	rm -f build/main