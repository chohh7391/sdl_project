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

//! One camera's view of one tag, expressed in base_link.
struct CameraObservation
{
    std::string camera;                                   //!< the camera frame
    geometry_msgs::msg::TransformStamped pose_in_base;    //!< base_link -> tag
    double distance = 0.0;                                //!< camera to tag [m]
};

struct ObjectData
{
    //! Rebuilt every fusion cycle: one entry per camera that saw the tag. This
    //! was a fixed pair of cam1/cam2 fields, which meant a third camera -- a
    //! wrist camera, say -- could not be added without touching the fusion
    //! itself. With two cameras the arithmetic below is unchanged.
    std::vector<CameraObservation> observations;

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

    //! Camera frames to fuse, in no particular order. Declared as a ROS
    //! parameter so a camera can be added from configuration rather than code.
    std::vector<std::string> camera_frames_;
    std::vector<std::string> target_objects_;
    std::map<std::string, std::string> object_tag_map_;
    // Translation from each object's TAG frame to the object's own frame, in the
    // tag frame. Read off the vessel assets: beaker.usd carries apriltag_00 and
    // flask.usd apriltag_01 at a local (0.15, 0, -0.062) from the vessel's
    // origin, i.e. a plate on the table 15 cm to the vessel's side and a vessel
    // half-height below its centre, with no rotation relative to the vessel.
    std::map<std::string, tf2::Vector3> tag_to_object_;
    std::map<std::string, ObjectData> objects_;
    FTData ft_data_;
    ScaleData scale_data_;
};