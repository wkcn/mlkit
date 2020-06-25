#include <boosted_tree/boosted_tree.h>
#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/io.h>
#include <boosted_tree/vec.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(boosted_tree, m) {
  py::class_<BoostedTreeParam>(m, "BoostedTreeParam")
      .def(py::init<>())
      .def_readwrite("max_depth", &BoostedTreeParam::max_depth)
      .def_readwrite("learning_rate", &BoostedTreeParam::learning_rate)
      .def_readwrite("n_estimators", &BoostedTreeParam::n_estimators)
      .def_readwrite("objective", &BoostedTreeParam::objective)
      .def_readwrite("reg_lambda", &BoostedTreeParam::reg_lambda)
      .def_readwrite("gamma", &BoostedTreeParam::gamma)
      .def_readwrite("n_jobs", &BoostedTreeParam::n_jobs)
      .def_readwrite("tree_method", &BoostedTreeParam::tree_method)
      .def_readwrite("sketch_eps", &BoostedTreeParam::sketch_eps)
      .def_readwrite("seed", &BoostedTreeParam::seed);

  py::class_<BoostedTree>(m, "BoostedTree")
      .def(py::init<const BoostedTreeParam &>())
      .def("train", &BoostedTree::train)
      .def("predict", &BoostedTree::predict);

  py::class_<CSRMatrix<float>>(m, "CSRMatrix").def(py::init<>());

  py::class_<Vec<float>>(m, "Vec", py::buffer_protocol())
      .def(py::init<>())
      .def_buffer([](Vec<float> &vec) -> py::buffer_info {
        return py::buffer_info(vec.data(), sizeof(float),
                               py::format_descriptor<float>::format(), 1,
                               {vec.size()}, {sizeof(float)});
      });

  m.def("ReadLibSVMFile", &ReadLibSVMFile<float, float>,
        py::return_value_policy::reference);
}
