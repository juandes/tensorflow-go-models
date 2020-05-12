// This is for testing the TensorFlow C library installation.
// To run it, execute make testC from the project's root directory

#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  return 0;
}