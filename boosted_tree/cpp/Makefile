debug:
	g++ src/main.cpp src/boosted_tree/boosted_tree.cpp src/boosted_tree/type_convert.cpp --std=c++17 -g -D MOBULA_LOG_DEBUG -fopenmp -lgtest -lpthread -I include -o main
release:
	g++ src/main.cpp src/boosted_tree/boosted_tree.cpp src/boosted_tree/type_convert.cpp --std=c++17 -O3 -fopenmp -lgtest -lpthread -I include -o main
test:
	g++ tests/test_main.cpp src/boosted_tree/boosted_tree.cpp src/boosted_tree/type_convert.cpp --std=c++17 -g -lpthread -lgtest -I include -o tests/test
	./tests/test
pythonlib:
	g++ src/boosted_tree/boosted_tree.cpp src/boosted_tree/type_convert.cpp --std=c++17 -O3 -fopenmp -lpthread -shared -I include -fPIC `python3 -m pybind11 --includes` python/boosted_tree/binding.cpp -o boosted_tree`python3-config --extension-suffix`
