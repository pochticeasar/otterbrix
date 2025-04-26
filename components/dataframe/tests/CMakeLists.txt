project(test_dataframe)

set(${PROJECT_NAME}_SOURCES
        bitmask/bitmask_tests.cpp
        column/test_column.cpp
        column/test_column_benchmark.cpp
        tools.cpp
)

add_executable(${PROJECT_NAME} main.cpp ${${PROJECT_NAME}_SOURCES})

target_link_libraries(
        ${PROJECT_NAME} PRIVATE
        otterbrix::dataframe
        Catch2::Catch2
        Boost::boost
        ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_definitions(${PROJECT_NAME} PRIVATE CATCH_CONFIG_ENABLE_BENCHMARKING)
include(CTest)
include(Catch)
catch_discover_tests(${PROJECT_NAME})
