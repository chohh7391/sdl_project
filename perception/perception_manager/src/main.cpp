#include "perception_manager/perception_manager.hpp"

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    
    auto node = std::make_shared<PerceptionManager>();
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}