#include "perception_manager/perception_manager.hpp"

using namespace std::chrono_literals;

PerceptionManager::PerceptionManager() : Node("perception_manager")
{
    RCLCPP_INFO(this->get_logger(), "Starting Perception Manager ...");

    // [설정] 추적할 대상 매핑
    target_objects_ = {"beaker", "flask"};
    object_tag_map_["beaker"] = "beaker_tag";
    object_tag_map_["flask"] = "flask_tag";

    auto update_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    
    // TF 초기화
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Publisher 생성 (Float32 적용)
    ft_pub_ = this->create_publisher<geometry_msgs::msg::Wrench>("/perception/filtered_ft", 10);
    scale_pub_ = this->create_publisher<std_msgs::msg::Float32>("/perception/fusion_scale", 10);

    // Timer 생성
    process_timer_ = this->create_wall_timer(
        20ms, std::bind(&PerceptionManager::process_raw_data, this));
    
    publish_timer_ = this->create_wall_timer(
        33ms, std::bind(&PerceptionManager::publish_processed_data, this));

    auto sub_opt = rclcpp::SubscriptionOptions();
    sub_opt.callback_group = update_cb_group;

    ft_sub_ = this->create_subscription<geometry_msgs::msg::Wrench>(
        "/raw_ft_data", 10, 
        [this](const geometry_msgs::msg::Wrench::SharedPtr msg) { this->update_raw_ft(msg); }, 
        sub_opt);
    scale_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/raw_scale_data", 10, 
        [this](const std_msgs::msg::Float32::SharedPtr msg) { this->update_raw_scale(msg); }, 
        sub_opt);

    RCLCPP_INFO(this->get_logger(), "Perception Manager Node has been started.");
}

void PerceptionManager::process_raw_data() {
    process_fusion_tf();
    process_raw_ft();
    process_raw_scale();
}

void PerceptionManager::publish_processed_data() {
    publish_processed_tf();
    publish_processed_ft();
    publish_processed_scale();
}

// ==========================================
// TF Fusion Logic (Offset 적용 추가됨)
// ==========================================
void PerceptionManager::process_fusion_tf() {
    for (const auto& obj_name : target_objects_) {
        std::string tag_id = object_tag_map_[obj_name];
        auto& obj = objects_[obj_name];
        
        obj.cam1_valid = false;
        obj.cam2_valid = false;

        // --- Camera 1 Lookup ---
        try {
            if (tf_buffer_->canTransform("base_link", "camera_1", tf2::TimePointZero) &&
                tf_buffer_->canTransform("camera_1", tag_id, tf2::TimePointZero)) 
            {
                 auto base_to_cam1 = tf_buffer_->lookupTransform("base_link", "camera_1", tf2::TimePointZero);
                 auto cam1_to_tag = tf_buffer_->lookupTransform("camera_1", tag_id, tf2::TimePointZero);

                 tf2::Transform t_base_cam1, t_cam1_tag;
                 tf2::fromMsg(base_to_cam1.transform, t_base_cam1);
                 tf2::fromMsg(cam1_to_tag.transform, t_cam1_tag);
                 tf2::Transform t_result = t_base_cam1 * t_cam1_tag;
                 
                 obj.cam1_raw_base.header.frame_id = "base_link"; 
                 obj.cam1_raw_base.header.stamp = cam1_to_tag.header.stamp;
                 obj.cam1_raw_base.child_frame_id = obj_name;
                 obj.cam1_raw_base.transform = tf2::toMsg(t_result);
                 
                 obj.dist_to_cam1 = std::sqrt(pow(cam1_to_tag.transform.translation.x, 2) +
                                              pow(cam1_to_tag.transform.translation.y, 2) +
                                              pow(cam1_to_tag.transform.translation.z, 2));
                 obj.cam1_valid = true;
            }
        } catch (const tf2::TransformException &ex) {
            RCLCPP_DEBUG(this->get_logger(), "Cam1 TF missing: %s", ex.what());
        }

        // --- Camera 2 Lookup ---
        try {
            if (tf_buffer_->canTransform("base_link", "camera_2", tf2::TimePointZero) &&
                tf_buffer_->canTransform("camera_2", tag_id, tf2::TimePointZero)) 
            {
                 auto base_to_cam2 = tf_buffer_->lookupTransform("base_link", "camera_2", tf2::TimePointZero);
                 auto cam2_to_tag = tf_buffer_->lookupTransform("camera_2", tag_id, tf2::TimePointZero);

                 tf2::Transform t_base_cam2, t_cam2_tag;
                 tf2::fromMsg(base_to_cam2.transform, t_base_cam2);
                 tf2::fromMsg(cam2_to_tag.transform, t_cam2_tag);
                 tf2::Transform t_result = t_base_cam2 * t_cam2_tag;

                 obj.cam2_raw_base.header.frame_id = "base_link";
                 obj.cam2_raw_base.header.stamp = cam2_to_tag.header.stamp;
                 obj.cam2_raw_base.child_frame_id = obj_name;
                 obj.cam2_raw_base.transform = tf2::toMsg(t_result);

                 obj.dist_to_cam2 = std::sqrt(pow(cam2_to_tag.transform.translation.x, 2) +
                                              pow(cam2_to_tag.transform.translation.y, 2) +
                                              pow(cam2_to_tag.transform.translation.z, 2));
                 obj.cam2_valid = true;
            }
        } catch (const tf2::TransformException &ex) {
            RCLCPP_DEBUG(this->get_logger(), "Cam2 TF missing: %s", ex.what());
        }

        // --- Data Fusion & Offset Application ---
        
        geometry_msgs::msg::TransformStamped final_tf_msg;
        bool has_valid_data = false;

        // 1. Tag 위치 융합 (Tag Pose Fusion)
        if (obj.cam1_valid && obj.cam2_valid) {
            // 시간: 더 최신 것 사용
            rclcpp::Time t1(obj.cam1_raw_base.header.stamp);
            rclcpp::Time t2(obj.cam2_raw_base.header.stamp);
            final_tf_msg.header.stamp = (t1 > t2) ? t1 : t2;

            double eps = 1e-6;
            double w1 = 1.0 / (obj.dist_to_cam1 * obj.dist_to_cam1 + eps);
            double w2 = 1.0 / (obj.dist_to_cam2 * obj.dist_to_cam2 + eps);
            double W1 = w1 / (w1 + w2);
            double W2 = w2 / (w1 + w2);

            auto& tr1 = obj.cam1_raw_base.transform.translation;
            auto& tr2 = obj.cam2_raw_base.transform.translation;
            
            final_tf_msg.transform.translation.x = (tr1.x * W1) + (tr2.x * W2);
            final_tf_msg.transform.translation.y = (tr1.y * W1) + (tr2.y * W2);
            final_tf_msg.transform.translation.z = (tr1.z * W1) + (tr2.z * W2);

            tf2::Quaternion q1, q2, q_fused;
            tf2::fromMsg(obj.cam1_raw_base.transform.rotation, q1);
            tf2::fromMsg(obj.cam2_raw_base.transform.rotation, q2);
            q_fused = q1.slerp(q2, W2); 
            final_tf_msg.transform.rotation = tf2::toMsg(q_fused);
            
            has_valid_data = true;
        } 
        else if (obj.cam1_valid) {
            final_tf_msg = obj.cam1_raw_base;
            has_valid_data = true;
        } 
        else if (obj.cam2_valid) {
            final_tf_msg = obj.cam2_raw_base;
            has_valid_data = true;
        }

        // 2. 실제 Object 위치로 오프셋 적용 (Offset Application)
        if (has_valid_data) {
            // 2-1. 융합된 Tag 위치를 TF2 객체로 변환
            tf2::Transform t_tag_fused;
            tf2::fromMsg(final_tf_msg.transform, t_tag_fused);

            // 2-2. Tag에서 Object로 가는 변환 정의 (Tag -> Object)
            // Marker가 Object 기준 (-0.15, 0, 0)에 있다면,
            // Object는 Marker 기준 (+0.15, 0, 0)에 있습니다.
            // (Base -> Tag -> Object)
            tf2::Transform t_tag_to_obj;
            t_tag_to_obj.setIdentity();
            t_tag_to_obj.setOrigin(tf2::Vector3(-0.15, 0.0, 0.0)); // X축으로 -0.15m 이동
            // 만약 회전도 다르다면 여기서 setRotation으로 설정

            // 2-3. 최종 Object 위치 계산
            tf2::Transform t_object_final = t_tag_fused * t_tag_to_obj;

            // 2-4. 결과 저장
            final_tf_msg.transform = tf2::toMsg(t_object_final);
            final_tf_msg.header.frame_id = "base_link";
            final_tf_msg.child_frame_id = obj_name; // 이제 이것은 Tag가 아닌 Object 중심입니다.
            
            obj.processed_data = final_tf_msg;
        }
    }
}

// ==========================================
// 기타 데이터 처리
// ==========================================
void PerceptionManager::update_raw_ft(const geometry_msgs::msg::Wrench::SharedPtr msg) {
    ft_data_.raw_data = *msg;
}

void PerceptionManager::update_raw_scale(const std_msgs::msg::Float32::SharedPtr msg) {
    scale_data_.raw_data = *msg;
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

// ==========================================
// 데이터 발행 (Publish)
// ==========================================
void PerceptionManager::publish_processed_tf() {
    for (const auto& obj_name : target_objects_) {
        auto& obj = objects_[obj_name];

        if (obj.cam1_valid || obj.cam2_valid) {
            if (obj.processed_data.header.frame_id.empty()) {
                obj.processed_data.header.frame_id = "base_link";
            }
            tf_broadcaster_->sendTransform(obj.processed_data);
        }
    }
}

void PerceptionManager::publish_processed_ft() {
    ft_pub_->publish(ft_data_.processed_data);
}

void PerceptionManager::publish_processed_scale() {
    scale_pub_->publish(scale_data_.processed_data);
}