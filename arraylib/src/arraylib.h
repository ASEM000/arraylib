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

#ifdef _OMP_H
#include <omp.h>
#endif

#define FREE(ptr) _Generic((ptr), NDArray *: array_free, Layout *: layout_free, default: free)(ptr)

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
    size_t size; /**< size of the memory block. */
    size_t refs; /**< Reference count for the memory block. */
} Data;

typedef struct {
    size_t ndim;    /**< Number of dimension of data. */
    size_t* shape;  /**< Shape of data. */
    size_t* stride; /**< Stride of data. */
} Layout;

/**
 * @brief NDArray structure representing an N-dimensional array.
 */
typedef struct {
    Data* data;  /**< Pointer to the data structure. */
    f32* ptr;    /**< Offseted pointer to the start of the data memory. */
    Layout* lay; /**< Memory layout (shape, stride). */
    bool view;   /**< Flag indicating if this is a view. */
} NDArray;

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

// -------------------------------------------------------------------------------------------------
// DATA
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates an empty Data structure with the given size.
 * @param size const size_t* of the memory block.
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

Layout* layout_copy(Layout* dst, const Layout* src);

void layout_free(Layout* lay);
Layout** layout_broadcast(const Layout** lays, size_t nlay);
NDArray* array_as_strided(const NDArray* src, const Layout* lay);
// -------------------------------------------------------------------------------------------------
// ARRAY CREATION AND DESTRUCTION
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates an empty contigouous NDArray with the given shape and dimensions.
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
NDArray* array_shallow_copy(const NDArray* array);

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
NDArray* array_deep_copy(const NDArray* array);

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
 * f32 value = array_get_elem(array, index); // Gets the value at index (1, 1)
 * @endcode
 */
f32 array_get_elem(const NDArray* array, const size_t* index);

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
 * NDArray* view = array_get_view(array, start, end, step); // Gets a view of the array
 * @endcode
 */
NDArray* array_get_view(
        const NDArray* array,
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
 * array_set_elem_from_scalar(array, 5.0, index); // Sets the value at index (1, 1) to 5.0
 * @endcode
 */
NDArray* array_set_elem_from_scalar(NDArray* array, f32 value, const size_t* index);

/**
 * @brief Sets a scalar value in an NDArray over a range of indices.
 * @param array Pointer to the NDArray.
 * @param value Value to set.
 * @param start Array of start indices.
 * @param end Array of end indices.
 * @param step Array of step sizes.
 * @return Pointer to the NDArray.
 */
NDArray* array_set_view_from_scalar(
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
        const NDArray* src,
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
NDArray* array_reshape(const NDArray* array, const size_t* shape, size_t ndim);

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
NDArray* array_transpose(const NDArray* array, const size_t* dst);

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
 * NDArray* moved = array_move_dim(array, src, dst, 2); // Swaps the axes
 * @endcode
 */
NDArray* array_move_dim(const NDArray* array, const size_t* src, const size_t* dst, size_t ndim);

/**
 * @brief Flattens an NDArray into a 1D array.
 * @param array Pointer to the NDArray.
 * @return Pointer to the flattened NDArray.
 *
 * @code
 * NDArray* flattened = array_ravel(array); // Flattens the array
 * @endcode
 */
NDArray* array_ravel(const NDArray* array);

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
NDArray* array_scalar_op(binop fn, const NDArray* lhs, f32 rhs);

NDArray* array_scalar_sum(const NDArray* src, f32 rhs);
NDArray* array_scalar_sub(const NDArray* src, f32 rhs);
NDArray* array_scalar_mul(const NDArray* src, f32 rhs);
NDArray* array_scalar_div(const NDArray* src, f32 rhs);
NDArray* array_scalar_mod(const NDArray* src, f32 rhs);
NDArray* array_scalar_pow(const NDArray* src, f32 rhs);
NDArray* array_scalar_max(const NDArray* src, f32 rhs);
NDArray* array_scalar_min(const NDArray* src, f32 rhs);
NDArray* array_scalar_eq(const NDArray* src, f32 rhs);
NDArray* array_scalar_ne(const NDArray* src, f32 rhs);
NDArray* array_scalar_gt(const NDArray* src, f32 rhs);
NDArray* array_scalar_ge(const NDArray* src, f32 rhs);
NDArray* array_scalar_lt(const NDArray* src, f32 rhs);
NDArray* array_scalar_le(const NDArray* src, f32 rhs);

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
NDArray* array_array_matmul(const NDArray* lhs, const NDArray* rhs);

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
 * NDArray* result = array_array_op(add, array1, array2); // Adds two arrays element-wise
 * @endcode
 */
NDArray* array_array_op(binop fn, const NDArray* lhs, const NDArray* rhs);

NDArray* array_array_sum(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_sub(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_mul(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_div(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_mod(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_pow(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_max(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_min(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_eq(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_ne(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_gt(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_ge(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_lt(const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_le(const NDArray* lhs, const NDArray* rhs);

// -------------------------------------------------------------------------------------------------
// ELEMENTWISE OPERATIONS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Apply fn to all items of NDArray.
 * @param fn Unary function to to apply.
 * @param array Pointer to the NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_op(uniop fn, const NDArray* array);

NDArray* array_neg(const NDArray* src);
NDArray* array_abs(const NDArray* src);
NDArray* array_sqrt(const NDArray* src);
NDArray* array_exp(const NDArray* src);
NDArray* array_log(const NDArray* src);
NDArray* array_sin(const NDArray* src);
NDArray* array_cos(const NDArray* src);
NDArray* array_tan(const NDArray* src);
NDArray* array_asin(const NDArray* src);
NDArray* array_acos(const NDArray* src);
NDArray* array_atan(const NDArray* src);
NDArray* array_sinh(const NDArray* src);
NDArray* array_cosh(const NDArray* src);
NDArray* array_tanh(const NDArray* src);
NDArray* array_asinh(const NDArray* src);
NDArray* array_acosh(const NDArray* src);
NDArray* array_atanh(const NDArray* src);
NDArray* array_ceil(const NDArray* src);
NDArray* array_floor(const NDArray* src);

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
NDArray* array_array_dot(const NDArray* lhs, const NDArray* rhs);

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
 * NDArray* result = array_reduce(add, array, axes, 1, 0.0); // Sums along the first dim
 * @endcode
 */
NDArray* array_reduce(
        binop acc_fn,
        const NDArray* src,
        const size_t* reduce_dims,
        size_t reduce_ndim,
        f32 acc_init);

NDArray* array_reduce_sum(const NDArray* src, const size_t* reduce_dims, size_t reduce_ndim);
NDArray* array_reduce_max(const NDArray* src, const size_t* reduce_dims, size_t reduce_ndim);
NDArray* array_reduce_min(const NDArray* src, const size_t* reduce_dims, size_t reduce_ndim);

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
NDArray* array_array_array_where(const NDArray* cond, const NDArray* lhs, const NDArray* rhs);
NDArray* array_array_scalar_where(const NDArray* cond, const NDArray* lhs, f32 rhs);
NDArray* array_scalar_array_where(const NDArray* cond, f32 lhs, const NDArray* rhs);
NDArray* array_scalar_scalar_where(const NDArray* cond, f32 lhs, f32 rhs);

// END

#endif  // ARRAYLIB_H