#pragma once

#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm> // std::find

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_msgs/msg/tf_message.hpp> // TF 메시지 직접 구독용
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/wrench.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>

struct ObjectData
{
    geometry_msgs::msg::TransformStamped cam1_raw_base; 
    geometry_msgs::msg::TransformStamped cam2_raw_base; 
    double dist_to_cam1; 
    double dist_to_cam2; 
    bool cam1_valid = false;
    bool cam2_valid = false;
    
    geometry_msgs::msg::TransformStamped processed_data; 
};

struct FTData { geometry_msgs::msg::Wrench raw_data; geometry_msgs::msg::Wrench processed_data; };
struct ScaleData { std_msgs::msg::Float64 raw_data; std_msgs::msg::Float64 processed_data; };

class PerceptionManager : public rclcpp::Node
{
public:
    PerceptionManager();

private:
    void process_raw_data();
    void publish_processed_data();

    // 콜백 함수들
    void tf_callback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void update_raw_ft(const geometry_msgs::msg::Wrench::SharedPtr msg);
    void update_raw_scale(const std_msgs::msg::Float64::SharedPtr msg);

    void process_raw_tf();
    void process_raw_ft();
    void process_raw_scale();

    void publish_processed_tf();
    void publish_processed_ft();
    void publish_processed_scale();

    rclcpp::TimerBase::SharedPtr process_timer_;
    rclcpp::TimerBase::SharedPtr publish_timer_;

    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr tf_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Wrench>::SharedPtr ft_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr scale_pub_;

    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_; // TF 직접 구독
    rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr ft_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr scale_sub_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    std::vector<std::string> target_objects_;
    std::map<std::string, ObjectData> objects_;
    FTData ft_data_;
    ScaleData scale_data_;
};