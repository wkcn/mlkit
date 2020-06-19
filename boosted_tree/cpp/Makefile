all:
	g++ src/main.cpp src/boosted_tree/boosted_tree.cpp --std=c++17 -g -fopenmp -lgtest -lpthread -I include -o main
release:
	g++ src/main.cpp src/boosted_tree/boosted_tree.cpp --std=c++17 -O3 -fopenmp -lgtest -lpthread -I include -o main
test:
	g++ tests/test_main.cpp --std=c++17 -g -lgtest -lpthread -I include -o tests/test
	./tests/test
