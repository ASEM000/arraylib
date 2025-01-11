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

/**
 * @brief NDArray structure representing an N-dimensional array.
 */
typedef struct {
    Data* data;      /**< Pointer to the data structure. */
    size_t* shape;   /**< Array of dimensions. */
    size_t* stride;  /**< Array of strides for each dimension. */
    size_t* bstride; /**< Array of back strides for each dimension. */
    size_t ndim;     /**< Number of dimensions. */
    size_t offset;   /**< Offset in the data memory. */
    bool view;       /**< Flag indicating if this is a view. */
} NDArray;

/**
 * @brief Iterator structure for iterating over an NDArray.
 */
typedef struct {
    f32* ptr;        /**< Pointer to the current element. */
    size_t* shape;   /**< Shape of the array being iterated. */
    size_t* stride;  /**< Strides of the array being iterated. */
    size_t* bstride; /**< Back strides of the array being iterated. */
    size_t* index;   /**< Current index in the iteration. */
    size_t* dims;    /**< Dimensions to iterate over. */
    size_t counter;  /**< Counter for the iteration. */
    size_t size;     /**< Total size of the iteration. */
    size_t ndim;     /**< Number of dimensions. */
} NDIterator;

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
 */
size_t prod(size_t* nums, size_t ndim);

/**
 * @brief Creates an array of size_t with the given number of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the created array.
 */
size_t* size_t_create(size_t ndim);

/**
 * @brief Sets all elements of an array to a given value.
 * @param dst Destination array.
 * @param value Value to set.
 * @param size Size of the array.
 * @return Pointer to the destination array.
 */
size_t* size_t_set(size_t* dst, size_t value, size_t size);

/**
 * @brief Copies the contents of one array to another.
 * @param dst Destination array.
 * @param src Source array.
 * @param size Size of the arrays.
 * @return Pointer to the destination array.
 */
size_t* size_t_copy(size_t* dst, size_t* src, size_t size);

/**
 * @brief Computes the flat index from a multi-dimensional index.
 * @param index Multi-dimensional index.
 * @param stride Strides for each dimension.
 * @param ndim Number of dimensions.
 * @return Flat index.
 */
size_t compute_flat_index(size_t* index, size_t* stride, size_t ndim);

/**
 * @brief Clamps a value between a minimum and maximum.
 * @param value Value to clamp.
 * @param minval Minimum value.
 * @param maxval Maximum value.
 * @return Clamped value.
 */
f32 clamp(f32 value, f32 minval, f32 maxval);

/**
 * @brief Checks if an array is contiguous in memory.
 * @param array Array to check.
 * @return True if the array is contiguous, false otherwise.
 */
bool is_contiguous(NDArray* array);

/**
 * @brief Computes the ceiling division of two numbers.
 * @param a Dividend.
 * @param b Divisor.
 * @return Ceiling division result.
 */
size_t cdiv(size_t a, size_t b);

/**
 * @brief Computes the strides for an array given its shape.
 * @param dst Destination array for strides.
 * @param shape Shape of the array.
 * @param ndim Number of dimensions.
 * @return Pointer to the destination array.
 */
size_t* compute_stride(size_t* dst, size_t* shape, size_t ndim);

/**
 * @brief Computes the back strides for an array given its shape and strides.
 * @param dst Destination array for back strides.
 * @param shape Shape of the array.
 * @param stride Strides of the array.
 * @param ndim Number of dimensions.
 * @return Pointer to the destination array.
 */
size_t* compute_bstride(size_t* dst, size_t* shape, size_t* stride, size_t ndim);

// -------------------------------------------------------------------------------------------------
// DATA
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates an empty Data structure with the given size.
 * @param size Size of the memory block.
 * @return Pointer to the created Data structure.
 */
Data* data_empty(size_t size);

// -------------------------------------------------------------------------------------------------
// ARRAY CREATION AND DESTRUCTION
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates an empty NDArray with the given shape and dimensions.
 * @param shape Array of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the created NDArray.
 */
NDArray* array_empty(size_t* shape, size_t ndim);

/**
 * @brief Frees the memory allocated for an NDArray.
 * @param array Pointer to the NDArray to free.
 */
void array_free(NDArray* array);

// -------------------------------------------------------------------------------------------------
// ITERATOR
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates an iterator for an NDArray.
 * @param ptr Pointer to the data.
 * @param shape Shape of the array.
 * @param stride Strides of the array.
 * @param bstride Back strides of the array.
 * @param dims Dimensions to iterate over.
 * @param ndim Number of dimensions.
 * @return The created iterator.
 */
NDIterator iter_create(
        f32* ptr,
        size_t* shape,
        size_t* stride,
        size_t* bstride,
        size_t* dims,
        size_t ndim);

/**
 * @brief Creates an iterator for an NDArray.
 * @param array Pointer to the NDArray.
 * @param dims Dimensions to iterate over.
 * @return The created iterator.
 */
NDIterator iter_array(NDArray* array, size_t* dims);

/**
 * @brief Frees the memory allocated for an iterator.
 * @param iterator Pointer to the iterator to free.
 */
void iter_free(NDIterator* iterator);

/**
 * @brief Advances the iterator to the next element.
 * @param iter Pointer to the iterator.
 * @return True if the iterator has more elements, false otherwise.
 */
bool iter_next(NDIterator* iter);

// -------------------------------------------------------------------------------------------------
// COPY
// -------------------------------------------------------------------------------------------------

/**
 * @brief Creates a shallow copy of an NDArray.
 * @param array Pointer to the NDArray to copy.
 * @return Pointer to the shallow copy.
 */
NDArray* array_shallow_copy(NDArray* array);

/**
 * @brief Creates a deep copy of an NDArray.
 * @param array Pointer to the NDArray to copy.
 * @return Pointer to the deep copy.
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
 */
NDArray* array_zeros(size_t* shape, size_t ndim);

/**
 * @brief Creates an NDArray filled with the given elements.
 * @param elems Array of elements to fill.
 * @param shape Array of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the created NDArray.
 */
NDArray* array_fill(f32* elems, size_t* shape, size_t ndim);

/**
 * @brief Creates an NDArray filled with ones.
 * @param shape Array of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the created NDArray.
 */
NDArray* array_ones(size_t* shape, size_t ndim);

/**
 * @brief Creates an NDArray with values ranging from start to end with a given step.
 * @param start Starting value.
 * @param end Ending value.
 * @param step Step size.
 * @return Pointer to the created NDArray.
 */
NDArray* array_arange(f32 start, f32 end, f32 step);

/**
 * @brief Creates an NDArray with values linearly spaced between start and end.
 * @param start Starting value.
 * @param end Ending value.
 * @param n Number of values.
 * @return Pointer to the created NDArray.
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
 */
f32 array_get_scalar_from_index(NDArray* array, size_t* index);

/**
 * @brief Gets a view of an NDArray from a range of indices.
 * @param array Pointer to the NDArray.
 * @param start Array of start indices.
 * @param end Array of end indices.
 * @param step Array of step sizes.
 * @return Pointer to the created view.
 */
NDArray* array_get_view_from_range(NDArray* array, size_t* start, size_t* end, size_t* step);

// -------------------------------------------------------------------------------------------------
// SETTERS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Sets a scalar value in an NDArray using an index.
 * @param array Pointer to the NDArray.
 * @param index Array of indices.
 * @param value Value to set.
 * @return Pointer to the NDArray.
 */
NDArray* array_set_scalar_from_index(NDArray* array, size_t* index, f32 value);

/**
 * @brief Sets a scalar value in an NDArray over a range of indices.
 * @param array Pointer to the NDArray.
 * @param start Array of start indices.
 * @param end Array of end indices.
 * @param step Array of step sizes.
 * @param value Value to set.
 * @return Pointer to the NDArray.
 */
NDArray* array_set_scalar_from_range(
        NDArray* array,
        size_t* start,
        size_t* end,
        size_t* step,
        f32 value);

/**
 * @brief Sets a view of an NDArray from another NDArray.
 * @param array Pointer to the NDArray.
 * @param start Array of start indices.
 * @param end Array of end indices.
 * @param step Array of step sizes.
 * @param value Pointer to the NDArray to set.
 * @return Pointer to the NDArray.
 */
NDArray* array_set_view_from_array(
        NDArray* array,
        size_t* start,
        size_t* end,
        size_t* step,
        NDArray* value);

// -------------------------------------------------------------------------------------------------
// RESHAPING
// -------------------------------------------------------------------------------------------------

/**
 * @brief Reshapes an NDArray to the given shape.
 * @param array Pointer to the NDArray.
 * @param shape Array of dimensions.
 * @param ndim Number of dimensions.
 * @return Pointer to the reshaped NDArray.
 */
NDArray* array_reshape(NDArray* array, size_t* shape, size_t ndim);

/**
 * @brief Transposes an NDArray according to the given destination axes.
 * @param array Pointer to the NDArray.
 * @param dst Array of destination axes.
 * @return Pointer to the transposed NDArray.
 */
NDArray* array_transpose(NDArray* array, size_t* dst);

/**
 * @brief Moves axes of an NDArray to new positions.
 * @param array Pointer to the NDArray.
 * @param src Array of source axes.
 * @param dst Array of destination axes.
 * @param ndim Number of dimensions.
 * @return Pointer to the NDArray with moved axes.
 */
NDArray* array_move_axis(NDArray* array, size_t* src, size_t* dst, size_t ndim);

/**
 * @brief Flattens an NDArray into a 1D array.
 * @param array Pointer to the NDArray.
 * @return Pointer to the flattened NDArray.
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
 */
NDArray* array_scalar_op(binop fn, NDArray* lhs, f32 rhs);

/**
 * @brief Adds a scalar to an NDArray.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_scalar_add(NDArray* lhs, f32 rhs);

/**
 * @brief Subtracts a scalar from an NDArray.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_scalar_sub(NDArray* lhs, f32 rhs);

/**
 * @brief Multiplies an NDArray by a scalar.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_scalar_mul(NDArray* lhs, f32 rhs);

/**
 * @brief Divides an NDArray by a scalar.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_scalar_div(NDArray* lhs, f32 rhs);

/**
 * @brief Raises an NDArray to the power of a scalar.
 * @param lhs Pointer to the NDArray.
 * @param rhs Scalar value.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_scalar_pow(NDArray* lhs, f32 rhs);

// -------------------------------------------------------------------------------------------------
// MATMUL
// -------------------------------------------------------------------------------------------------

/**
 * @brief Performs matrix multiplication between two NDArrays.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
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
 */
NDArray* array_array_scalar_op(binop fn, NDArray* lhs, NDArray* rhs);

/**
 * @brief Adds two NDArrays element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_sum(NDArray* lhs, NDArray* rhs);

/**
 * @brief Subtracts two NDArrays element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_sub(NDArray* lhs, NDArray* rhs);

/**
 * @brief Multiplies two NDArrays element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_mul(NDArray* lhs, NDArray* rhs);

/**
 * @brief Divides two NDArrays element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_div(NDArray* lhs, NDArray* rhs);

/**
 * @brief Raises one NDArray to the power of another element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_pow(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if two NDArrays are equal element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_eq(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if two NDArrays are not equal element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_neq(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if the left-hand side NDArray is greater than the right-hand side element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_gt(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if the left-hand side NDArray is greater than or equal to the right-hand side
 * element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_geq(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if the left-hand side NDArray is less than the right-hand side element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_lt(NDArray* lhs, NDArray* rhs);

/**
 * @brief Checks if the left-hand side NDArray is less than or equal to the right-hand side
 * element-wise.
 * @param lhs Pointer to the left-hand side NDArray.
 * @param rhs Pointer to the right-hand side NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_array_leq(NDArray* lhs, NDArray* rhs);

// -------------------------------------------------------------------------------------------------
// ELEMENTWISE OPERATIONS
// -------------------------------------------------------------------------------------------------

/**
 * @brief Computes the natural logarithm of an NDArray element-wise.
 * @param array Pointer to the NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_log(NDArray* array);

/**
 * @brief Negates an NDArray element-wise.
 * @param array Pointer to the NDArray.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_neg(NDArray* array);

/**
 * @brief Computes the exponential of an NDArray element-wise.
 * @param array Pointer to the NDArray.
 * @return Pointer to the resulting NDArray.
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
 */
NDArray* array_reduce(binop acc_fn, NDArray* array, size_t* axes, size_t num_axes, f32 acc_init);

/**
 * @brief Computes the maximum value of an NDArray along specified dimensions.
 * @param array Pointer to the NDArray.
 * @param reduce_dims Array of dimensions to reduce.
 * @param ndim Number of dimensions.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_reduce_max(NDArray* array, size_t* reduce_dims, size_t ndim);

/**
 * @brief Computes the minimum value of an NDArray along specified dimensions.
 * @param array Pointer to the NDArray.
 * @param reduce_dims Array of dimensions to reduce.
 * @param ndim Number of dimensions.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_reduce_min(NDArray* array, size_t* reduce_dims, size_t ndim);

/**
 * @brief Computes the sum of an NDArray along specified dimensions.
 * @param array Pointer to the NDArray.
 * @param reduce_dims Array of dimensions to reduce.
 * @param ndim Number of dimensions.
 * @return Pointer to the resulting NDArray.
 */
NDArray* array_reduce_sum(NDArray* array, size_t* reduce_dims, size_t ndim);

#endif  // ARRAYLIB_H