# CMake generated Testfile for 
# Source directory: C:/Users/User/OneDrive/Projects/Banyan/C++/bayes_tree
# Build directory: C:/Users/User/OneDrive/Projects/Banyan/C++/bayes_tree/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test_tree]=] "C:/Users/User/OneDrive/Projects/Banyan/C++/bayes_tree/build/test_tree.exe")
set_tests_properties([=[test_tree]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/User/OneDrive/Projects/Banyan/C++/bayes_tree/CMakeLists.txt;38;add_test;C:/Users/User/OneDrive/Projects/Banyan/C++/bayes_tree/CMakeLists.txt;0;")
add_test([=[test_categorical_distribution]=] "C:/Users/User/OneDrive/Projects/Banyan/C++/bayes_tree/build/test_categorical_distribution.exe")
set_tests_properties([=[test_categorical_distribution]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/User/OneDrive/Projects/Banyan/C++/bayes_tree/CMakeLists.txt;39;add_test;C:/Users/User/OneDrive/Projects/Banyan/C++/bayes_tree/CMakeLists.txt;0;")
subdirs("python")
