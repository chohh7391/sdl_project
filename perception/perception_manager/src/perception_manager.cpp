#include "perception_manager/perception_manager.hpp"

using namespace std::chrono_literals;

PerceptionManager::PerceptionManager() : Node("perception_manager")
{
    RCLCPP_INFO(this->get_logger(), "Starting Perception Manager ...");

    // 추적할 대상 지정
    target_objects_ = {"beaker", "flask"};

    auto update_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto process_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto publish_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    tf_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/perception/fused_pose", 10);
    ft_pub_ = this->create_publisher<geometry_msgs::msg::Wrench>("/perception/filtered_ft", 10);
    scale_pub_ = this->create_publisher<std_msgs::msg::Float64>("/perception/fusion_scale", 10);

    // TF가 콜백으로 들어오므로 update_timer_는 삭제하여 최적화했습니다.
    process_timer_ = this->create_wall_timer(
        20ms, std::bind(&PerceptionManager::process_raw_data, this));
    publish_timer_ = this->create_wall_timer(
        33ms, std::bind(&PerceptionManager::publish_processed_data, this));

    auto sub_opt = rclcpp::SubscriptionOptions();
    sub_opt.callback_group = update_cb_group;

    // TF 토픽 직접 구독 (AprilTag의 데이터를 가로챔)
    tf_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", 10, 
        [this](const tf2_msgs::msg::TFMessage::SharedPtr msg) { this->tf_callback(msg); }, 
        sub_opt);
    ft_sub_ = this->create_subscription<geometry_msgs::msg::Wrench>(
        "/ft_data", 10, 
        [this](const geometry_msgs::msg::Wrench::SharedPtr msg) { this->update_raw_ft(msg); }, 
        sub_opt);
    scale_sub_ = this->create_subscription<std_msgs::msg::Float64>(
        "/scale_data", 10, 
        [this](const std_msgs::msg::Float64::SharedPtr msg) { this->update_raw_scale(msg); }, 
        sub_opt);

    RCLCPP_INFO(this->get_logger(), "Perception Manager Node has been started.");
}

void PerceptionManager::process_raw_data() {
    process_raw_tf();
    process_raw_ft();
    process_raw_scale();
}

void PerceptionManager::publish_processed_data() {
    RCLCPP_INFO(this->get_logger(), "no response");
    publish_processed_tf();
    publish_processed_ft();
    publish_processed_scale();
}

// ==========================================
// TF 가로채기 (핵심 로직)
// ==========================================
void PerceptionManager::tf_callback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    for (const auto& transform : msg->transforms) {
        std::string child = transform.child_frame_id;
        std::string parent = transform.header.frame_id;

        if (std::find(target_objects_.begin(), target_objects_.end(), child) != target_objects_.end()) {
            // [디버그] 어떤 데이터가 들어오는지 확인
            // RCLCPP_INFO(this->get_logger(), "Detected: %s from %s", child.c_str(), parent.c_str());

            auto& obj = objects_[child];
            try {
                // 이 부분이 실패하면 cam1_valid가 절대 true가 되지 않습니다.
                auto base_to_cam = tf_buffer_->lookupTransform("base_link", parent, tf2::TimePointZero);
                
                tf2::Transform tf_base_to_cam, tf_cam_to_obj;
                tf2::fromMsg(base_to_cam.transform, tf_base_to_cam);
                tf2::fromMsg(transform.transform, tf_cam_to_obj);
                
                tf2::Transform tf_base_to_obj = tf_base_to_cam * tf_cam_to_obj;
                
                geometry_msgs::msg::TransformStamped base_to_obj_msg;
                base_to_obj_msg.header.stamp = transform.header.stamp;
                base_to_obj_msg.header.frame_id = "base_link";
                base_to_obj_msg.child_frame_id = child + "_fused";
                base_to_obj_msg.transform = tf2::toMsg(tf_base_to_obj);

                double dist = std::sqrt(std::pow(transform.transform.translation.x, 2) + 
                                        std::pow(transform.transform.translation.y, 2) + 
                                        std::pow(transform.transform.translation.z, 2));

                if (parent == "camera_1") {
                    obj.cam1_raw_base = base_to_obj_msg;
                    obj.dist_to_cam1 = dist;
                    obj.cam1_valid = true;
                } else if (parent == "camera_2") {
                    obj.cam2_raw_base = base_to_obj_msg;
                    obj.dist_to_cam2 = dist;
                    obj.cam2_valid = true;
                }
            } catch (const tf2::TransformException & ex) {
                // 여기서 에러가 난다면 base_link <-> camera_1 사이의 TF가 없는 것입니다.
                RCLCPP_ERROR(this->get_logger(), "TF Error in callback: %s", ex.what());
            }
        }
    }
}

void PerceptionManager::update_raw_ft(const geometry_msgs::msg::Wrench::SharedPtr msg) {
    ft_data_.raw_data = *msg;
}
void PerceptionManager::update_raw_scale(const std_msgs::msg::Float64::SharedPtr msg) {
    scale_data_.raw_data = *msg;
}

// ==========================================
// 센서 퓨전 (타임아웃 및 가중치)
// ==========================================
void PerceptionManager::process_raw_tf() {
    // rclcpp::Time now = this->get_clock()->now();
    // double timeout = 0.5; // 0.5초 이상 안 보이면 유효하지 않음 처리

    for (const auto& obj_name : target_objects_) {
        auto& obj = objects_[obj_name];
        
        // Timeout 검사 (로봇의 안전한 조작을 위해 필수)
        // if (obj.cam1_valid) {
        //     rclcpp::Time cam1_time(obj.cam1_raw_base.header.stamp);
        //     if ((now - cam1_time).seconds() > timeout) obj.cam1_valid = false;
        // }
        // if (obj.cam2_valid) {
        //     rclcpp::Time cam2_time(obj.cam2_raw_base.header.stamp);
        //     if ((now - cam2_time).seconds() > timeout) obj.cam2_valid = false;
        // }

        geometry_msgs::msg::TransformStamped fused_tf;
        // fused_tf.header.stamp = now;
        fused_tf.header.frame_id = "base_link";
        fused_tf.child_frame_id = obj_name + "_fused";

        // 케이스 1: 두 카메라 모두 물체를 볼 때
        if (obj.cam1_valid && obj.cam2_valid) {
            double eps = 1e-6;
            double w1 = 1.0 / (obj.dist_to_cam1 * obj.dist_to_cam1 + eps);
            double w2 = 1.0 / (obj.dist_to_cam2 * obj.dist_to_cam2 + eps);
            
            double W1 = w1 / (w1 + w2);
            double W2 = w2 / (w1 + w2);

            auto& t1 = obj.cam1_raw_base.transform.translation;
            auto& t2 = obj.cam2_raw_base.transform.translation;
            fused_tf.transform.translation.x = (t1.x * W1) + (t2.x * W2);
            fused_tf.transform.translation.y = (t1.y * W1) + (t2.y * W2);
            fused_tf.transform.translation.z = (t1.z * W1) + (t2.z * W2);

            tf2::Quaternion q1, q2, q_fused;
            tf2::fromMsg(obj.cam1_raw_base.transform.rotation, q1);
            tf2::fromMsg(obj.cam2_raw_base.transform.rotation, q2);
            q_fused = q1.slerp(q2, W2);
            fused_tf.transform.rotation = tf2::toMsg(q_fused);

            obj.processed_data = fused_tf;
        } 
        // 케이스 2: 한 대의 카메라만 볼 때
        else if (obj.cam1_valid) {
            obj.processed_data = obj.cam1_raw_base;
            obj.processed_data.child_frame_id = obj_name + "_fused";
        } 
        else if (obj.cam2_valid) {
            obj.processed_data = obj.cam2_raw_base;
            obj.processed_data.child_frame_id = obj_name + "_fused";
        }
    }
}

void PerceptionManager::process_raw_ft() {
    double alpha = 0.2;
    auto& raw = ft_data_.raw_data.force;
    auto& proc = ft_data_.processed_data.force;
    proc.x = alpha * raw.x + (1.0 - alpha) * proc.x;
    proc.y = alpha * raw.y + (1.0 - alpha) * proc.y;
    proc.z = alpha * raw.z + (1.0 - alpha) * proc.z;

    auto& raw_t = ft_data_.raw_data.torque;
    auto& proc_t = ft_data_.processed_data.torque;
    proc_t.x = alpha * raw_t.x + (1.0 - alpha) * proc_t.x;
    proc_t.y = alpha * raw_t.y + (1.0 - alpha) * proc_t.y;
    proc_t.z = alpha * raw_t.z + (1.0 - alpha) * proc_t.z;
}

void PerceptionManager::process_raw_scale() {
    scale_data_.processed_data = scale_data_.raw_data;
}

void PerceptionManager::publish_processed_tf() {
    for (const auto& obj_name : target_objects_) {
        auto& obj = objects_[obj_name];

        RCLCPP_INFO(this->get_logger(), "Object: %s | Cam1: %d | Cam2: %d", 
                    obj_name.c_str(), obj.cam1_valid, obj.cam2_valid);
        
        // [디버그 로그 추가]
        if (!obj.cam1_valid && !obj.cam2_valid) {
            // 이 로그가 계속 뜬다면 데이터가 valid로 판정되지 못하고 있는 겁니다.
            RCLCPP_DEBUG(this->get_logger(), "Object %s is not valid yet.", obj_name.c_str());
        }

        if (obj.cam1_valid || obj.cam2_valid) {
            tf_pub_->publish(obj.processed_data);
        }
    }
}

void PerceptionManager::publish_processed_ft() {
    ft_pub_->publish(ft_data_.processed_data);
}

void PerceptionManager::publish_processed_scale() {
    scale_pub_->publish(scale_data_.processed_data);
}