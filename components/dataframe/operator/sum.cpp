#include <boost/compute.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/command_queue.hpp>
#include <memory_resource>
#include "core/uvector.hpp"
#include "core/buffer.hpp"
#include "../column/column.hpp"
namespace compute = boost::compute;

float hz() {
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);
    
    compute::vector<int> ids(context);
    compute::vector<float> values(context);

    core::uvector<int> uvec2(std::pmr::get_default_resource(), 100);
    std::iota(uvec2.begin(), uvec2.end(), 1);
    column_t col2(resource, std::move(uvec2), core::buffer(), 0);
    auto view2 = co12.view();
    auto* data2 = view2.data<int>();

    ids = compute::vector<int>(host_ids.begin(), host_ids.end(), queue);
    values = compute::vector<float>(host_values.begin(), host_values.end(), queue);

    float sum = compute::accumulate(
        values.begin(), values.end(), 0.0f, compute::plus<float>(), queue
    );
    return sum;
}