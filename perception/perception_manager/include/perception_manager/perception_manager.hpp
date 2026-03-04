#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h> 
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/wrench.hpp>
#include <std_msgs/msg/float32.hpp> // Float32로 변경됨
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

struct ObjectData
{
    geometry_msgs::msg::TransformStamped cam1_raw_base; 
    geometry_msgs::msg::TransformStamped cam2_raw_base; 
    double dist_to_cam1; 
    double dist_to_cam2; 
    bool cam1_valid = false;
    bool cam2_valid = false;
    
    // 최종 가공된 데이터 (Offset 적용 후)
    geometry_msgs::msg::TransformStamped processed_data; 
};

struct FTData { geometry_msgs::msg::Wrench raw_data; geometry_msgs::msg::Wrench processed_data; };
struct ScaleData { std_msgs::msg::Float32 raw_data; std_msgs::msg::Float32 processed_data; };

class PerceptionManager : public rclcpp::Node
{
public:
    PerceptionManager();

private:
    void process_raw_data();
    void publish_processed_data();

    void update_raw_ft(const geometry_msgs::msg::Wrench::SharedPtr msg);
    void update_raw_scale(const std_msgs::msg::Float32::SharedPtr msg);

    void process_fusion_tf();
    void process_raw_ft();
    void process_raw_scale();

    void publish_processed_tf();
    void publish_processed_ft();
    void publish_processed_scale();

    rclcpp::TimerBase::SharedPtr process_timer_;
    rclcpp::TimerBase::SharedPtr publish_timer_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    rclcpp::Publisher<geometry_msgs::msg::Wrench>::SharedPtr ft_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr scale_pub_;

    rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr ft_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr scale_sub_;

    std::vector<std::string> target_objects_;
    std::map<std::string, std::string> object_tag_map_;
    std::map<std::string, ObjectData> objects_;
    FTData ft_data_;
    ScaleData scale_data_;
};