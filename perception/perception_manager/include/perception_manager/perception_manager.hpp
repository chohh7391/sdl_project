#ifndef PERCEPTION_MANAGER_HPP
#define PERCEPTION_MANAGER_HPP

#include "rclcpp/rclcpp.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/wrench.hpp"

#include "perception_interfaces/srv/get_object_info.hpp" // 사용자 정의 서비스 메시지
#include "perception_interfaces/srv/get_ft_data.hpp"

#include <string>
#include <map>
#include "rclcpp/executors/multi_threaded_executor.hpp"


// 오브젝트 정보를 저장할 구조체
struct ObjectData
{
    std::string name;
    std::string frame_id;

    geometry_msgs::msg::Pose grasping_offset;
    geometry_msgs::msg::TransformStamped current_transform;
};

class PerceptionManager : public rclcpp::Node
{
public:
    PerceptionManager();

private:
    void timer_callback();
    void get_object_info_callback(
        const std::shared_ptr<perception_interfaces::srv::GetObjectInfo::Request> request,
        const std::shared_ptr<perception_interfaces::srv::GetObjectInfo::Response> response);
    // F/T 센서 데이터를 받을 콜백 함수
    void ft_sensor_callback(const geometry_msgs::msg::Wrench::SharedPtr msg);
    
    // F/T 센서 데이터를 제공할 서비스 콜백 함수
    void get_ft_data_callback(
        const std::shared_ptr<perception_interfaces::srv::GetFtData::Request> request,
        const std::shared_ptr<perception_interfaces::srv::GetFtData::Response> response);

    // Configuration
    std::map<int, std::string> object_id_to_name_map_;
    geometry_msgs::msg::Wrench latest_ft_data_;

    // Latest Data
    std::map<int, ObjectData> object_data_map_; // key: AprilTag/QR ID
    std::vector<geometry_msgs::msg::Pose> grasping_offsets_;

    // TF
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    // Services
    rclcpp::Service<perception_interfaces::srv::GetObjectInfo>::SharedPtr get_object_info_service_;
    rclcpp::Service<perception_interfaces::srv::GetFtData>::SharedPtr get_ft_data_service_;

    // Force Torque Sensor Subscription
    rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr ft_sensor_subscriber_;

    // Timer
    rclcpp::TimerBase::SharedPtr object_info_timer_;
};

#endif // PERCEPTION_MANAGER_HPP