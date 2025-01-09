#include "arraylib.h"
#include <setjmp.h>
#include <signal.h>

jmp_buf jump_buffer;
int assertion_failures = 0;

void handle_sigabrt(int signum) {
  longjmp(jump_buffer, 1); // Jump back to the setjmp point
}

#define TEST(expr)                                                             \
  signal(SIGABRT, handle_sigabrt);                                             \
  if (setjmp(jump_buffer) == 0) {                                              \
    expr;                                                                      \
  } else {                                                                     \
    assertion_failures++;                                                      \
  }

void test_alloc() {
  void *ptr = alloc(10);
  TEST(assert(ptr != NULL));
}

void test_utils() {
  size_t nums1[] = {1, 2, 3};
  TEST(assert(prod(nums1, 3) == 6));

  //  +1
  size_t nums2[] = {1, 2, 3};
  TEST(prod(nums2, 0));

  size_t *arr = size_t_create(5);
  TEST(assert(arr != NULL));
  free(arr);

  size_t arr2[5];
  TEST(assert(size_t_set(arr2, 42, 5) == arr2));
  TEST(assert(arr2[0] == 42 && arr2[4] == 42));

  size_t src[] = {1, 2, 3};
  size_t dst[3];
  TEST(assert(size_t_copy(dst, src, 3) == dst));
  TEST(assert(dst[0] == 1 && dst[2] == 3));

  size_t index[] = {1, 1};
  size_t stride[] = {3, 1};
  TEST(assert(flat_index(index, stride, 2) == 4));

  TEST(assert(clamp(5, 0, 10) == 5));
  TEST(assert(clamp(-1, 0, 10) == 0));
  TEST(assert(clamp(11, 0, 10) == 10));
}

void test_array_creation() {
  size_t shape[] = {2, 3};
  NDArray *array = array_empty(shape, 2);
  TEST(assert(array != NULL));
  TEST(assert(array->ndim == 2));
  TEST(assert(array->shape[0] == 2 && array->shape[1] == 3));
  array_free(array);

  // + 1
  TEST(array_free(NULL));
}

void test_iterator() {
  size_t shape[] = {2, 3};
  NDArray *array = array_empty(shape, 2);
  NDIterator iter = array_iter(array);
  TEST(assert(iter.ptr != NULL));
  TEST(assert(iter.size == 6));

  size_t count = 0;
  while (iterator_iterate(&iter)) {
    count++;
  }
  TEST(assert(count == 6));

  NDIterator iter_axis = array_iter_axis(array, 0, 1);
  TEST(assert(iter_axis.ptr != NULL));
  TEST(assert(iter_axis.size == 3));

  array_free(array);
}

void test_array_initialization() {
  size_t shape[] = {2, 2};
  NDArray *zeros = array_zeros(shape, 2);
  TEST(assert(zeros != NULL));
  TEST(assert(zeros->data->mem[0] == 0 && zeros->data->mem[3] == 0));
  array_free(zeros);

  f32 elems[] = {1, 2, 3, 4};
  NDArray *filled = array_fill(elems, shape, 2);
  TEST(assert(filled != NULL));
  TEST(assert(filled->data->mem[0] == 1 && filled->data->mem[3] == 4));
  array_free(filled);

  NDArray *ones = array_ones(shape, 2);
  TEST(assert(ones != NULL));
  TEST(assert(ones->data->mem[0] == 1 && ones->data->mem[3] == 1));
  array_free(ones);

  NDArray *arange = array_arange(0, 5, 1);
  TEST(assert(arange != NULL));
  TEST(assert(arange->data->mem[0] == 0 && arange->data->mem[4] == 4));
  array_free(arange);

  NDArray *linspace = array_linspace(0, 1, 5);
  TEST(assert(linspace != NULL));
  TEST(assert(linspace->data->mem[0] == 0 && linspace->data->mem[4] == 1));
  array_free(linspace);
}

void test_array_operations() {
  size_t shape[] = {2, 2};
  NDArray *array = array_ones(shape, 2);
  NDArray *result = array_scalar_add(array, 1);
  TEST(assert(result != NULL));
  TEST(assert(result->data->mem[0] == 2 && result->data->mem[3] == 2));
  array_free(array);
  array_free(result);

  NDArray *array1 = array_ones(shape, 2);
  NDArray *array2 = array_ones(shape, 2);
  NDArray *sum = array_array_add(array1, array2);
  TEST(assert(sum != NULL));
  TEST(assert(sum->data->mem[0] == 2 && sum->data->mem[3] == 2));
  array_free(array1);
  array_free(array2);
  array_free(sum);
}

void test_reduction_operations() {
  size_t shape[] = {3};
  NDArray *array1 = array_fill((f32[]){1, 2, 3}, shape, 1);
  NDArray *array2 = array_fill((f32[]){4, 5, 6}, shape, 1);
  NDArray *dot = array_array_dot(array1, array2);
  TEST(assert(dot != NULL));
  TEST(assert(dot->data->mem[0] == 32));
  array_free(array1);
  array_free(array2);
  array_free(dot);
}

void test_view_from_range() {
  size_t shape[] = {3, 3};
  NDArray *array = array_empty(shape, 2);

  for (size_t i = 0; i < 9; i++) {
    array->data->mem[i] = (f32)i;
  }

  size_t start[] = {1, 1};
  size_t end[] = {3, 3};
  size_t step[] = {1, 1};
  NDArray *view = array_get_view_from_range(array, start, end, step);

  TEST(assert(view->shape[0] == 2 && view->shape[1] == 2));
  TEST(assert(view->data->mem[view->offset + 0] == 4.0f)); // [1, 1]
  TEST(assert(view->data->mem[view->offset + 1] == 5.0f)); // [1, 2]
  TEST(assert(view->data->mem[view->offset + 3] == 7.0f)); // [2, 1]
  TEST(assert(view->data->mem[view->offset + 4] == 8.0f)); // [2, 2]

  // Free the view and original array
  array_free(view);
  array_free(array);
}

void test_modify_view() {
  size_t shape[] = {3, 3};
  NDArray *array = array_empty(shape, 2);

  memset(array->data->mem, 0, sizeof(f32) * 9);

  size_t start[] = {0, 0};
  size_t end[] = {1, 3};
  size_t step[] = {1, 1};
  NDArray *view = array_get_view_from_range(array, start, end, step);

  array_set_scalar_from_index(view, (size_t[]){0, 1}, 42.0f);

  TEST(assert(array->data->mem[1] == 42.0f));

  array_free(view);
  array_free(array);
}

void test_non_contiguous_view() {
  size_t shape[] = {4};
  NDArray *array = array_empty(shape, 1);

  for (size_t i = 0; i < 4; i++) {
    array->data->mem[i] = (f32)i;
  }

  size_t start[] = {0};
  size_t end[] = {4};
  size_t step[] = {2};
  NDArray *view = array_get_view_from_range(array, start, end, step);

  TEST(assert(view->shape[0] == 2));
  TEST(assert(view->data->mem[view->offset + 0] == 0.0f)); // [0]
  TEST(assert(view->data->mem[view->offset + 2] == 2.0f)); // [2]

  array_free(view);
  array_free(array);
}

void test_basic_matmul() {
  size_t shape[] = {2, 2};
  NDArray *lhs = array_empty(shape, 2);
  NDArray *rhs = array_empty(shape, 2);

  lhs->data->mem[0] = 1.0f;
  lhs->data->mem[1] = 2.0f;
  lhs->data->mem[2] = 3.0f;
  lhs->data->mem[3] = 4.0f;

  rhs->data->mem[0] = 5.0f;
  rhs->data->mem[1] = 6.0f;
  rhs->data->mem[2] = 7.0f;
  rhs->data->mem[3] = 8.0f;
  NDArray *result = array_array_matmul(lhs, rhs);

  TEST(assert(result != NULL));
  TEST(assert(result->shape[0] == 2 && result->shape[1] == 2));
  TEST(assert(result->data->mem[0] == 19.0f)); // [1*5 + 2*7]
  TEST(assert(result->data->mem[1] == 22.0f)); // [1*6 + 2*8]
  TEST(assert(result->data->mem[2] == 43.0f)); // [3*5 + 4*7]
  TEST(assert(result->data->mem[3] == 50.0f)); // [3*6 + 4*8]

  array_free(lhs);
  array_free(rhs);
  array_free(result);
}

void test_dimension_mismatch() {
  size_t shape1[] = {2, 3};
  size_t shape2[] = {4, 2};
  NDArray *lhs = array_empty(shape1, 2);
  NDArray *rhs = array_empty(shape2, 2);

  //  + 1
  TEST(array_array_matmul(lhs, rhs));

  array_free(lhs);
  array_free(rhs);
}

void test_matmul_identity() {
  size_t shape[] = {2, 2};
  NDArray *matrix = array_empty(shape, 2);
  matrix->data->mem[0] = 1.0f;
  matrix->data->mem[1] = 2.0f;
  matrix->data->mem[2] = 3.0f;
  matrix->data->mem[3] = 4.0f;

  NDArray *identity = array_empty(shape, 2);
  identity->data->mem[0] = 1.0f;
  identity->data->mem[1] = 0.0f;
  identity->data->mem[2] = 0.0f;
  identity->data->mem[3] = 1.0f;

  NDArray *result = array_array_matmul(matrix, identity);

  TEST(assert(result != NULL));
  TEST(assert(result->shape[0] == 2 && result->shape[1] == 2));
  TEST(assert(result->data->mem[0] == 1.0f));
  TEST(assert(result->data->mem[1] == 2.0f));
  TEST(assert(result->data->mem[2] == 3.0f));
  TEST(assert(result->data->mem[3] == 4.0f));

  // Free the matrices
  array_free(matrix);
  array_free(identity);
  array_free(result);
}

void test_blocked_matmul() {
  size_t shape[] = {4, 4};
  NDArray *lhs = array_empty(shape, 2);
  NDArray *rhs = array_empty(shape, 2);

  for (size_t i = 0; i < 16; i++) {
    lhs->data->mem[i] = (f32)i;
  }

  for (size_t i = 0; i < 16; i++) {
    rhs->data->mem[i] = (f32)i;
  }

  NDArray *result = array_array_matmul(lhs, rhs);

  TEST(assert(result != NULL));
  TEST(assert(result->shape[0] == 4 && result->shape[1] == 4));

  f32 expected[] = {56.0f,  62.0f,  68.0f,  74.0f,  152.0f, 174.0f,
                    196.0f, 218.0f, 248.0f, 286.0f, 324.0f, 362.0f,
                    344.0f, 398.0f, 452.0f, 506.0f};

  for (size_t i = 0; i < 16; i++) {
    TEST(assert(result->data->mem[i] == expected[i]));
  }

  array_free(lhs);
  array_free(rhs);
  array_free(result);
}

int main() {
  test_alloc();
  test_utils();
  test_array_creation();
  test_iterator();
  test_array_initialization();
  test_array_operations();
  test_reduction_operations();
  test_view_from_range();
  test_modify_view();
  test_non_contiguous_view();
  test_basic_matmul();
  test_matmul_identity();
  test_blocked_matmul();
  assert(assertion_failures == 2);
  printf("tests passed %d.", assertion_failures);
  return 0;
}