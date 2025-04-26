#pragma once
#include <boost/compute.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/command_queue.hpp>
#include <cstdint>
#include <memory_resource>
#include "core/uvector.hpp"
#include "core/buffer.hpp"
#include "../column/column.hpp"
namespace compute = boost::compute;

template <typename T>
T sum(const components::dataframe::column::column_t& column, size_t size) {
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);
    
    compute::vector<T> values(context);
    values.resize(size);

    auto view2 = column.view();
    auto* data2 = view2.data<T>();

    compute::copy(data2, data2 +size, values.begin(), queue);

    T sum = 0;
    compute::reduce(
        values.begin(), values.end(), &sum, compute::plus<T>(), queue
    );
    return sum;
}