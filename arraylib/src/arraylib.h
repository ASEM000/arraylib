/**
 * @file arraylib.h
 * @brief Simple ND-Array Implementation
 * @author Mahmoud Asem
 * @date 2025-01-01
 */

#ifndef ARRAYLIB_H
#define ARRAYLIB_H

#define BLOCK_SIZE 32

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ITERDIM SIZE_MAX
#define ITERALL ((DimSpecs){.nspec = 0})

#define FREE(ptr)                  \
    _Generic(                      \
            (ptr),                 \
            NDArray *: array_free, \
            NDIter *: iter_free,   \
            Layout *: layout_free, \
            default: free)(ptr)

// START

// -------------------------------------------------------------------------------------------------
// STRUCTS/TYPEDEFS
// -------------------------------------------------------------------------------------------------

/**
 * @brief 32-bit floating point type.
 */
typedef float f32;

/**
 * @brief Binary operation function pointer type.
 * @param lhs Left-hand side operand.
 * @param rhs Right-hand side operand.
 * @return Result of the binary operation.
 */
typedef f32 (*binop)(f32, f32);

/**
 * @brief Unary operation function pointer type.
 * @param lhs Operand.
 * @return Result of the unary operation.
 */
typedef f32 (*uniop)(f32);

/**
 * @brief Data structure holding the memory and metadata for an array.
 */
typedef struct {
    f32* mem;    /**< Pointer to the memory block. */
    size_t size; /**< Size of the memory block. */
    size_t refs; /**< Reference count for the memory block. */
} Data;

typedef struct {
    size_t* shape;  /**< Shape of data. */
    size_t* stride; /**< Stride of data. */
    size_t ndim;    /**< Number of dimension of data. */
} Layout;

typedef struct {
    Layout* lhs;
    Layout* rhs;
} LayoutPair;

/**
 * @brief NDArray structure representing an N-dimensional array.
 */
typedef struct {
    Data* data;  /**< Pointer to the data structure. */
    f32* ptr;    /**< Offseted pointer to the start of the data memory. */
    Layout* lay; /**< Memory layout (shape, stride). */
    bool view;   /**< Flag indicating if this is a view. */
} NDArray;

/**
 * @brief Iterator structure for iterating over an NDArray.
 */
typedef struct {
    size_t dim;
    size_t value;
} DimSpec;

typedef struct {
    DimSpec* specs;
    size_t nspec;
} DimSpecs;

typedef struct {
    f32* ptr;       /**< Pointer to the current element. */
    Layout* lay;    /**< Pointer to the memory layout (shape, stride). */
    size_t* index;  /**< Current index in the iteration. */
    size_t* dims;   /**< Dimensions to iterate over. */
    size_t counter; /**< Counter for the iteration. */
    size_t size;    /**< Total size of the iteration. */
} NDIter;

// -------------------------------------------------------------------------------------------------
// FUNCTION DECLARATIONS
// -------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------------
// MEMORY ALLOCATORS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Allocates memory of the given size.
 * @param size Size of the memory to allocate.
 * @return Pointer to the allocated memory.
 *
 * @code
 * void* memory = alloc(100 * sizeof(int)); // Allocates memory for 100 integers
 * @endcode
 */
void* alloc(size_t size);

// -------------------------------------------------------------------------------------------------
// UTILS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Computes the product of an array of numbers.
 * @param nums Array of numbers.
 * @param ndim Number of dimensions.
 * @return Product of the numbers.
 *
 * @code
 * size_t nums[] = {2, 3, 4};
 * size_t result = prod(nums, 3); // result = 24
 * @endcode
 */
size_t prod(const size_t* nums, size_t ndim);

/**
 * @brief Creates an array of size_t with the given number of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the created array.
 *
 * @code
 * size_t* arr = size_t_alloc(3); // Creates an array of size 3
 * @endcode
 */
size_t* size_t_alloc(size_t ndim);

/**
 * @brief Sets all elements of an array to a given value.
 * @param dst Destination array.
 * @param value Value to set.
 * @param size Size of the array.
 * @return Pointer to the destination array.
 *
 * @code
 * size_t arr[3];
 * size_t_set(arr, 5, 3); // arr = {5, 5, 5}
 * @endcode
 */
size_t* size_t_set(size_t* dst, size_t value, size_t size);

/**
 * @brief Copies the contents of one array to another.
 * @param dst Destination array.
 * @param src Source array.
 * @param size Size of the arrays.
 * @return Pointer to the destination array.
 *
 * @code
 * size_t src[] = {1, 2, 3};
 * size_t dst[3];
 * size_t_copy(dst, src, 3); // dst = {1, 2, 3}
 * @endcode
 */
size_t* size_t_copy(size_t* dst, const size_t* src, size_t size);

/**
 * @brief Computes the flat index from a multi-dimensional index.
 * @param index Multi-dimensional index.
 * @param stride Strides for each dimension.
 * @param ndim Number of dimensions.
 * @return Flat index.
 *
 * @code
 * size_t index[] = {1, 1};
 * size_t stride[] = {3, 1};
 * size_t flat_idx = compute_flat_index(index, stride, 2); // flat_idx = 4
 * @endcode
 */
size_t compute_flat_index(const size_t* index, const size_t* stride, size_t ndim);

/**
 * @brief Clamps a value between a minimum and maximum.
 * @param value Value to clamp.
 * @param minval Minimum value.
 * @param maxval Maximum value.
 * @return Clamped value.
 *
 * @code
 * f32 result = clamp(10.5, 0.0, 10.0); // result = 10.0
 * @endcode
 */
f32 clamp(f32 value, f32 minval, f32 maxval);

/**
 * @brief Checks if an array is contiguous in memory.
 * @param array Array to check.
 * @return True if the array is contiguous, false otherwise.
 *
 * @code
 * NDArray* array = array_zeros((size_t[]){2, 2}, 2);
 * bool contiguous = is_contiguous(array); // contiguous = true
 * @endcode
 */
bool is_contiguous(const NDArray* array);

bool is_broadcastable(const Layout* lhs, const Layout* rhs);

/**
 * @brief Computes the ceiling division of two numbers.
 * @param a Dividend.
 * @param b Divisor.
 * @return Ceiling division result.
 *
 * @code
 * size_t result = cdiv(10, 3); // result = 4
 * @endcode
 */
size_t cdiv(size_t a, size_t b);

/**
 * @brief Computes the strides for an array given its shape.
 * @param dst Destination array for strides.
 * @param shape Shape of the array.
 * @param ndim Number of dimensions.
 * @return Pointer to the destination array.
 *
 * @code
 * size_t shape[] = {2, 3};
 * size_t strides[2];
 * compute_stride(strides, shape, 2); // strides = {3, 1}
 * @endcode
 */
size_t* compute_stride(size_t* dst, const size_t* shape, const size_t ndim);

// -------------------------------------------------------------------------------------------------
// DATA
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates an empty Data structure with the given size.
 * @param size Size of the memory block.
 * @return Pointer to the created Data structure.
 *
 * @code
 * Data* data = data_empty(100); // Creates a Data structure with 100 elements
 * @endcode
 */
Data* data_empty(size_t size);

// -------------------------------------------------------------------------------------------------
// LAYOUT
// -------------------------------------------------------------------------------------------------

Layout* layout_alloc(size_t ndim);

Layout* layout_copy(const Layout* src);

void layout_free(Layout* lay);

LayoutPair* layout_broadcast(const LayoutPair* lay);

// -------------------------------------------------------------------------------------------------
// ARRAY CREATION AND DESTRUCTION
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates an empty NDArray with the given shape and dimensions.
 * @param shape Array of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the created NDArray.
 *
 * @code
 * size_t shape[] = {2, 3};
 * NDArray* array = array_empty(shape, 2); // Creates a 2x3 empty array
 * @endcode
 */
NDArray* array_empty(const size_t* shape, size_t ndim);

/**
 * @brief Frees the memory allocated for an NDArray.
 * @param array Pointer to the NDArray to free.
 *
 * @code
 * NDArray* array = array_zeros((size_t[]){2, 2}, 2);
 * array_free(array); // Frees the memory
 * @endcode
 */
void array_free(NDArray* array);

// -------------------------------------------------------------------------------------------------
// ITERATOR
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates an iterator for an NDArray.
 * @param ptr Pointer to the data.
 * @param lay Data layout (shape+stride).
 * @param specs Dimension spec.
 * @return Pointer to the created iterator.
 *
 * @code
 * NDArray* array = array_zeros((size_t[]){2, 2}, 2);
 * NDIter iter = iter_create(array->data->mem, array->shape, array->stride, array->bstride, 2,
 * ITERALL);
 * @endcode
 */
NDIter* iter_create(f32* ptr, const Layout* lay, const DimSpecs specs);

/**
 * @brief Frees the memory allocated for an iterator.
 * @param iterator Pointer to the iterator to free.
 *
 * @code
 * NDIter iter = iter_array(array, ITERALL);
 * iter_free(&iter); // Frees the iterator
 * @endcode
 */
void iter_free(NDIter* iterator);

/**
 * @brief Advances the iterator to the next element.
 * @param iter Pointer to the iterator.
 * @return True if the iterator has more elements, false otherwise.
 *
 * @code
 * NDIter iter = iter_array(array, ITERALL);
 * while (iter_next(iter)) {
 *     printf("%f ", *iter.ptr); // Prints each element
 * }
 * @endcode
 */
bool iter_next(NDIter* iter);

// -------------------------------------------------------------------------------------------------
// COPY
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates a shallow copy of an NDArray.
 * @param array Pointer to the NDArray to copy.
 * @return Pointer to the shallow copy.
 *
 * @code
 * NDArray* array = array_zeros((size_t[]){2, 2}, 2);
 * NDArray* copy = array_shallow_copy(array); // Creates a shallow copy
 * @endcode
 */
NDArray* array_shallow_copy(NDArray* array);

/**
 * @brief Creates a deep copy of an NDArray.
 * @param array Pointer to the NDArray to copy.
 * @return Pointer to the deep copy.
 *
 * @code
 * NDArray* array = array_zeros((size_t[]){2, 2}, 2);
 * NDArray* copy = array_deep_copy(array); // Creates a deep copy
 * @endcode
 */
NDArray* array_deep_copy(NDArray* array);

// -------------------------------------------------------------------------------------------------
// INITIALIZATION
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates an NDArray filled with zeros.
 * @param shape Array of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the created NDArray.
 *
 * @code
 * size_t shape[] = {2, 3};
 * NDArray* array = array_zeros(shape, 2); // Creates a 2x3 array filled with zeros
 * @endcode
 */
NDArray* array_zeros(const size_t* shape, size_t ndim);

/**
 * @brief Creates an NDArray filled with the given elements.
 * @param elems Array of elements to fill.
 * @param shape Array of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the created NDArray.
 *
 * @code
 * f32 elems[] = {1.0, 2.0, 3.0, 4.0};
 * size_t shape[] = {2, 2};
 * NDArray* array = array_fill(elems, shape, 2); // Creates a 2x2 array with the given elements
 * @endcode
 */
NDArray* array_fill(f32* elems, const size_t* shape, size_t ndim);

/**
 * @brief Creates an NDArray filled with ones.
 * @param shape Array of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the created NDArray.
 *
 * @code
 * size_t shape[] = {2, 2};
 * NDArray* array = array_ones(shape, 2); // Creates a 2x2 array filled with ones
 * @endcode
 */
NDArray* array_ones(const size_t* shape, size_t ndim);

/**
 * @brief Creates an NDArray with values ranging from start to end with a given step.
 * @param start Starting value.
 * @param end Ending value.
 * @param step Step size.
 * @return Pointer to the created NDArray.
 *
 * @code
 * NDArray* array = array_arange(0.0, 5.0, 1.0); // Creates an array [0.0, 1.0, 2.0, 3.0, 4.0]
 * @endcode
 */
NDArray* array_arange(f32 start, f32 end, f32 step);

/**
 * @brief Creates an NDArray with values linearly spaced between start and end.
 * @param start Starting value.
 * @param end Ending value.
 * @param n Number of values.
 * @return Pointer to the created NDArray.
 *
 * @code
 * NDArray* array = array_linspace(0.0, 1.0, 5); // Creates an array [0.0, 0.25, 0.5, 0.75, 1.0]
 * @endcode
 */
NDArray* array_linspace(f32 start, f32 end, f32 n);

// -------------------------------------------------------------------------------------------------
// GETTERS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Gets a scalar value from an NDArray using an index.
 * @param array Pointer to the NDArray.
 * @param index Array of indices.
 * @return The scalar value.
 *
 * @code
 * size_t index[] = {1, 1};
 * f32 value = array_get_scalar_from_index(array, index); // Gets the value at index (1, 1)
 * @endcode
 */
f32 array_get_scalar_from_index(NDArray* array, const size_t* index);

/**
 * @brief Gets a view of an NDArray from a range of indices.
 * @param array Pointer to the NDArray.
 * @param start Array of start indices.
 * @param end Array of end indices.
 * @param step Array of step sizes.
 * @return Pointer to the created view.
 *
 * @code
 * size_t start[] = {0, 0};
 * size_t end[] = {2, 2};
 * size_t step[] = {1, 1};
 * NDArray* view = array_get_view_from_range(array, start, end, step); // Gets a view of the array
 * @endcode
 */
NDArray* array_get_view_from_range(
        NDArray* array,
        const size_t* start,
        const size_t* end,
        const size_t* step);

// -------------------------------------------------------------------------------------------------
// SETTERS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Sets a scalar value in an NDArray using an index.
 * @param array Pointer to the NDArray.
 * @param index Array of indices.
 * @param value Value to set.
 * @return Pointer to the NDArray.
 *
 * @code
 * size_t index[] = {1, 1};
 * array_set_scalar_from_index(array, 5.0, index); // Sets the value at index (1, 1) to 5.0
 * @endcode
 */
NDArray* array_set_scalar_from_index(NDArray* array, f32 value, const size_t* index);

/**
 * @brief Sets a scalar value in an NDArray over a range of indices.
 * @param array Pointer to the NDArray.
 * @param value Value to set.
 * @param start Array of start indices.
 * @param end Array of end indices.
 * @param step Array of step sizes.
 * @return Pointer to the NDArray.
 */
NDArray* array_set_scalar_from_range(
        NDArray* array,
        f32 value,
        const size_t* start,
        const size_t* end,
        const size_t* step);

/**
 * @brief Sets a view of an NDArray from another NDArray.
 * @param dst Pointer to the NDArray.
 * @param src Pointer to the NDArray to set to destination.
 * @param start Array of start indices.
 * @param end Array of end indices.
 * @param step Array of step sizes.
 * @return Pointer to the NDArray.
 */
NDArray* array_set_view_from_array(
        NDArray* dst,
        NDArray* src,
        const size_t* start,
        const size_t* end,
        const size_t* step);

// -------------------------------------------------------------------------------------------------
// RESHAPING
// -------------------------------------------------------------------------------------------------

/**
 * @brief Reshapes an NDArray to the given shape.
 * @param array Pointer to the NDArray.
 * @param shape Array of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the reshaped NDArray.
 *
 * @code
 * size_t new_shape[] = {4};
 * NDArray* reshaped = array_reshape(array, new_shape, 1); // Reshapes the array to a 1D array of
 * size 4
 * @endcode
 */
NDArray* array_reshape(NDArray* array, const size_t* shape, size_t ndim);

/**
 * @brief Transposes an NDArray according to the given destination axes.
 * @param array Pointer to the NDArray.
 * @param dst Array of destination axes.
 * @return Pointer to the transposed NDArray.
 *
 * @code
 * size_t dst[] = {1, 0};
 * NDArray* transposed = array_transpose(array, dst); // Transposes the array
 * @endcode
 */
NDArray* array_transpose(NDArray* array, const size_t* dst);

/**
 * @brief Moves axes of an NDArray to new positions.
 * @param array Pointer to the NDArray.
 * @param src Array of source axes.
 * @param dst Array of destination axes.
 * @param ndim Number of dimensions.
 * @return Pointer to the NDArray with moved axes.
 *
 * @code
 * size_t src[] = {0, 1};
 * size_t dst[] = {1, 0};
 * NDArray* moved = array_move_axis(array, src, dst, 2); // Swaps the axes
 * @endcode
 */
NDArray* array_move_axis(NDArray* array, const size_t* src, const size_t* dst, size_t ndim);

/**
 * @brief Flattens an NDArray into a 1D array.
 * @param array Pointer to the NDArray.
 * @return Pointer to the flattened NDArray.
 *
 * @code
 * NDArray* flattened = array_ravel(array); // Flattens the array
 * @endcode
 */
NDArray* array_ravel(NDArray* array);

// -------------------------------------------------------------------------------------------------
// ARRAY-SCALAR OPERATIONS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Performs a binary operation between an NDArray and a scalar.
 * @param fn Binary operation function.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_scalar_op(add, array, 5.0); // Adds 5.0 to each element
 * @endcode
 */
NDArray* array_scalar_op(binop fn, NDArray* lhs, f32 rhs);

/**
 * @brief Adds a scalar to an NDArray.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_scalar_add(array, 5.0); // Adds 5.0 to each element
 * @endcode
 */
NDArray* array_scalar_add(NDArray* lhs, f32 rhs);

/**
 * @brief Subtracts a scalar from an NDArray.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_scalar_sub(array, 5.0); // Subtracts 5.0 from each element
 * @endcode
 */
NDArray* array_scalar_sub(NDArray* lhs, f32 rhs);

/**
 * @brief Multiplies an NDArray by a scalar.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_scalar_mul(array, 5.0); // Multiplies each element by 5.0
 * @endcode
 */
NDArray* array_scalar_mul(NDArray* lhs, f32 rhs);

/**
 * @brief Divides an NDArray by a scalar.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_scalar_div(array, 5.0); // Divides each element by 5.0
 * @endcode
 */
NDArray* array_scalar_div(NDArray* lhs, f32 rhs);

/**
 * @brief Raises an NDArray to the power of a scalar.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_scalar_pow(array, 2.0); // Squares each element
 * @endcode
 */
NDArray* array_scalar_pow(NDArray* lhs, f32 rhs);

NDArray* array_scalar_eq(NDArray* lhs, f32 rhs);
NDArray* array_scalar_neq(NDArray* lhs, f32 rhs);
NDArray* array_scalar_lt(NDArray* lhs, f32 rhs);
NDArray* array_scalar_leq(NDArray* lhs, f32 rhs);
NDArray* array_scalar_gt(NDArray* lhs, f32 rhs);
NDArray* array_scalar_geq(NDArray* lhs, f32 rhs);

// -------------------------------------------------------------------------------------------------
// MATMUL
// -------------------------------------------------------------------------------------------------

/**
 * @brief Performs matrix multiplication between two NDArrays.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_matmul(array1, array2); // Performs matrix multiplication
 * @endcode
 */
NDArray* array_array_matmul(NDArray* lhs, NDArray* rhs);

// -------------------------------------------------------------------------------------------------
// ARRAY-ARRAY OPERATIONS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Performs a binary operation between two NDArrays.
 * @param fn Binary operation function.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_scalar_op(add, array1, array2); // Adds two arrays element-wise
 * @endcode
 */
NDArray* array_array_scalar_op(binop fn, NDArray* lhs, NDArray* rhs);

/**
 * @brief Adds two NDArrays element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_sum(array1, array2); // Adds two arrays element-wise
 * @endcode
 */
NDArray* array_array_sum(NDArray* lhs, NDArray* rhs);

/**
 * @brief Subtracts two NDArrays element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_sub(array1, array2); // Subtracts two arrays element-wise
 * @endcode
 */
NDArray* array_array_sub(NDArray* lhs, NDArray* rhs);

/**
 * @brief Multiplies two NDArrays element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_mul(array1, array2); // Multiplies two arrays element-wise
 * @endcode
 */
NDArray* array_array_mul(NDArray* lhs, NDArray* rhs);

/**
 * @brief Divides two NDArrays element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_div(array1, array2); // Divides two arrays element-wise
 * @endcode
 */
NDArray* array_array_div(NDArray* lhs, NDArray* rhs);

/**
 * @brief Raises one NDArray to the power of another element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_pow(array1, array2); // Raises array1 to the power of array2
 * @endcode
 */
NDArray* array_array_pow(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if two NDArrays are equal element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_eq(array1, array2); // Checks for element-wise equality
 * @endcode
 */
NDArray* array_array_eq(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if two NDArrays are not equal element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_neq(array1, array2); // Checks for element-wise inequality
 * @endcode
 */
NDArray* array_array_neq(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if the left-hand side NDArray is greater than the right-hand side element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_gt(array1, array2); // Checks if array1 > array2 element-wise
 * @endcode
 */
NDArray* array_array_gt(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if the left-hand side NDArray is greater than or equal to the right-hand side
 * element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_geq(array1, array2); // Checks if array1 >= array2 element-wise
 * @endcode
 */
NDArray* array_array_geq(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if the left-hand side NDArray is less than the right-hand side element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_lt(array1, array2); // Checks if array1 < array2 element-wise
 * @endcode
 */
NDArray* array_array_lt(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if the left-hand side NDArray is less than or equal to the right-hand side
 * element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_leq(array1, array2); // Checks if array1 <= array2 element-wise
 * @endcode
 */
NDArray* array_array_leq(NDArray* lhs, NDArray* rhs);

// -------------------------------------------------------------------------------------------------
// ELEMENTWISE OPERATIONS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Apply fn to all items of NDArray.
 * @param fn Unary function to to apply.
 * @param array Pointer to the NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_op(uniop fn, NDArray* array);

/**
 * @brief Computes the natural logarithm of an NDArray element-wise.
 * @param array Pointer to the NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_log(array); // Computes the natural log of each element
 * @endcode
 */
NDArray* array_log(NDArray* array);

/**
 * @brief Negates an NDArray element-wise.
 * @param array Pointer to the NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_neg(array); // Negates each element
 * @endcode
 */
NDArray* array_neg(NDArray* array);

/**
 * @brief Computes the exponential of an NDArray element-wise.
 * @param array Pointer to the NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_exp(array); // Computes the exponential of each element
 * @endcode
 */
NDArray* array_exp(NDArray* array);

// -------------------------------------------------------------------------------------------------
// REDUCTION OPERATIONS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Computes the dot product of two NDArrays.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * NDArray* result = array_array_dot(array1, array2); // Computes the dot product
 * @endcode
 */
NDArray* array_array_dot(NDArray* lhs, NDArray* rhs);

/**
 * @brief Reduces an NDArray along specified dimensions using a binary operation.
 * @param acc_fn Binary operation function.
 * @param array Pointer to the NDArray.
 * @param axes Array of axes to reduce.
 * @param num_axes Number of axes.
 * @param acc_init Initial value for the accumulator.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * size_t axes[] = {0};
 * NDArray* result = array_reduce(add, array, axes, 1, 0.0); // Sums along the first axis
 * @endcode
 */
NDArray* array_reduce(
        binop acc_fn,
        NDArray* src,
        const size_t* reduce_dims,
        size_t reduce_ndim,
        f32 acc_init);
/**
 * @brief Computes the maximum value of an NDArray along specified dimensions.
 * @param array Pointer to the NDArray.
 * @param reduce_dims Array of dimensions to reduce.
 * @param ndim Number of dimensions.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * size_t reduce_dims[] = {0};
 * NDArray* result = array_reduce_max(array, reduce_dims, 1); // Finds the max along the first axis
 * @endcode
 */
NDArray* array_reduce_max(NDArray* array, size_t* reduce_dims, size_t ndim);

/**
 * @brief Computes the minimum value of an NDArray along specified dimensions.
 * @param array Pointer to the NDArray.
 * @param reduce_dims Array of dimensions to reduce.
 * @param ndim Number of dimensions.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * size_t reduce_dims[] = {0};
 * NDArray* result = array_reduce_min(array, reduce_dims, 1); // Finds the min along the first axis
 * @endcode
 */
NDArray* array_reduce_min(NDArray* array, size_t* reduce_dims, size_t ndim);

/**
 * @brief Computes the sum of an NDArray along specified dimensions.
 * @param array Pointer to the NDArray.
 * @param reduce_dims Array of dimensions to reduce.
 * @param ndim Number of dimensions.
 * @return Pointer to the resulting NDArray.
 *
 * @code
 * size_t reduce_dims[] = {0};
 * NDArray* result = array_reduce_sum(array, reduce_dims, 1); // Sums along the first axis
 * @endcode
 */
NDArray* array_reduce_sum(NDArray* array, size_t* reduce_dims, size_t ndim);

// -------------------------------------------------------------------------------------------------
// CONDITIONAL OPERATIONS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Return elements on the left/right NDArray based on condition.
 * @param cond Pointer to the conditional NDArray.
 * @param lhs Pointer to the on true NDArray.
 * @param rhs Pointer to the on false NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_where(NDArray* cond, NDArray* lhs, NDArray* rhs);

// END

#endif  // ARRAYLIB_H