#include "perception_manager/perception_manager.hpp"

#include <cstdlib>

using namespace std::chrono_literals;

PerceptionManager::PerceptionManager() : Node("perception_manager")
{
    RCLCPP_INFO(this->get_logger(), "Starting Perception Manager ...");

    // [설정] 추적할 대상 매핑
    // Which camera frames to fuse. A parameter so a wrist camera can be added
    // from configuration; the default is the cell's two fixed cameras, which is
    // what every measurement so far used.
    camera_frames_ = this->declare_parameter<std::vector<std::string>>(
        "camera_frames", std::vector<std::string>{"camera_1", "camera_2"});
    {
        std::string joined;
        for (const auto& c : camera_frames_) { joined += (joined.empty() ? "" : ", ") + c; }
        RCLCPP_INFO(this->get_logger(), "fusing %zu camera(s): %s",
                    camera_frames_.size(), joined.c_str());
    }

    target_objects_ = {"beaker", "flask"};
    object_tag_map_["beaker"] = "beaker_tag";
    object_tag_map_["flask"] = "flask_tag";

    // Tag -> object, in the tag's own frame. The tag is a plate on the table
    // 0.15 m to the vessel's side (so the vessel is at the tag's -x) and the
    // vessel's CENTRE is above the plate. Values are the local offsets authored
    // in beaker.usd / flask.usd; verified against the simulator's ground truth,
    // which put the reported tag 0.151 m from the vessel at the vessel's yaw.
    // The flask's plate is authored coplanar with the table and so never
    // rendered; task.py lifts it to 5 mm, which shortens this z by the lift
    // (0.0601 -> 0.0550). Keep this in step with SDL_TAG_Z_MIN.
    // With SDL_TAG_MOUNT=raise (the default) the plate sits on a post at
    // 0.18 m so neighbouring glassware cannot cover it; the vessel centre is
    // then BELOW the tag. beaker centre 0.0675 m, flask centre 0.0600 m.
    // SDL_TAG_Z overrides the mount height if task.py's is changed.
    const double tag_z = std::getenv("SDL_TAG_Z")
                             ? std::atof(std::getenv("SDL_TAG_Z")) : 0.18;
    tag_to_object_["beaker"] = tf2::Vector3(-0.15, 0.0, 0.0675 - tag_z);
    tag_to_object_["flask"] = tf2::Vector3(-0.15, 0.0, 0.0600 - tag_z);

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

        // Collect one observation per camera that can see the tag right now.
        obj.observations.clear();
        for (const auto& cam : camera_frames_) {
            try {
                if (!tf_buffer_->canTransform("base_link", cam, tf2::TimePointZero) ||
                    !tf_buffer_->canTransform(cam, tag_id, tf2::TimePointZero)) {
                    continue;
                }
                auto base_to_cam = tf_buffer_->lookupTransform("base_link", cam, tf2::TimePointZero);
                auto cam_to_tag = tf_buffer_->lookupTransform(cam, tag_id, tf2::TimePointZero);

                tf2::Transform t_base_cam, t_cam_tag;
                tf2::fromMsg(base_to_cam.transform, t_base_cam);
                tf2::fromMsg(cam_to_tag.transform, t_cam_tag);

                CameraObservation ob;
                ob.camera = cam;
                ob.pose_in_base.header.frame_id = "base_link";
                ob.pose_in_base.header.stamp = cam_to_tag.header.stamp;
                ob.pose_in_base.child_frame_id = obj_name;
                ob.pose_in_base.transform = tf2::toMsg(t_base_cam * t_cam_tag);
                ob.distance = std::sqrt(
                    pow(cam_to_tag.transform.translation.x, 2) +
                    pow(cam_to_tag.transform.translation.y, 2) +
                    pow(cam_to_tag.transform.translation.z, 2));
                obj.observations.push_back(ob);
            } catch (const tf2::TransformException &ex) {
                RCLCPP_DEBUG(this->get_logger(), "%s TF missing: %s", cam.c_str(), ex.what());
            }
        }

        // --- Data Fusion & Offset Application ---

        geometry_msgs::msg::TransformStamped final_tf_msg;
        bool has_valid_data = !obj.observations.empty();

        // 1. Tag 위치 융합 (Tag Pose Fusion)
        // Range-weighted over however many cameras saw the tag: a nearer view
        // is a better one, so each is weighted by 1/d^2. Rotation is blended by
        // successive slerp with the running weight, which for two cameras is
        // exactly q1.slerp(q2, w2/(w1+w2)) -- the two-camera result is bit for
        // bit what it was before this became a loop.
        if (obj.observations.size() == 1) {
            final_tf_msg = obj.observations.front().pose_in_base;
        } else if (obj.observations.size() > 1) {
            const double eps = 1e-6;
            std::vector<double> w;
            double w_sum = 0.0;
            for (const auto& ob : obj.observations) {
                double wi = 1.0 / (ob.distance * ob.distance + eps);
                w.push_back(wi);
                w_sum += wi;
            }

            // 시간: 가장 최신 관측 사용
            rclcpp::Time newest(obj.observations.front().pose_in_base.header.stamp);
            for (const auto& ob : obj.observations) {
                rclcpp::Time t(ob.pose_in_base.header.stamp);
                if (t > newest) { newest = t; }
            }
            final_tf_msg.header.stamp = newest;

            double x = 0.0, y = 0.0, z = 0.0;
            for (size_t i = 0; i < obj.observations.size(); ++i) {
                const auto& tr = obj.observations[i].pose_in_base.transform.translation;
                const double wi = w[i] / w_sum;
                x += tr.x * wi;
                y += tr.y * wi;
                z += tr.z * wi;
            }
            final_tf_msg.transform.translation.x = x;
            final_tf_msg.transform.translation.y = y;
            final_tf_msg.transform.translation.z = z;

            tf2::Quaternion q_fused;
            tf2::fromMsg(obj.observations.front().pose_in_base.transform.rotation, q_fused);
            double w_run = w[0];
            for (size_t i = 1; i < obj.observations.size(); ++i) {
                tf2::Quaternion qi;
                tf2::fromMsg(obj.observations[i].pose_in_base.transform.rotation, qi);
                q_fused = q_fused.slerp(qi, w[i] / (w_run + w[i]));
                w_run += w[i];
            }
            final_tf_msg.transform.rotation = tf2::toMsg(q_fused);
        }

        // 2. 실제 Object 위치로 오프셋 적용 (Offset Application)
        if (has_valid_data) {
            // 2-1. 융합된 Tag 위치를 TF2 객체로 변환
            tf2::Transform t_tag_fused;
            tf2::fromMsg(final_tf_msg.transform, t_tag_fused);

            // 2-2. Tag -> Object. Per object, because the plate sits a vessel
            // half-height below the vessel's centre and the two vessels differ
            // (see tag_to_object_). The z term was previously 0, which left
            // every reported object pose a half-height too low.
            tf2::Transform t_tag_to_obj;
            t_tag_to_obj.setIdentity();
            auto offset_it = tag_to_object_.find(obj_name);
            if (offset_it == tag_to_object_.end()) {
                RCLCPP_WARN_ONCE(this->get_logger(),
                    "No tag->object offset for '%s'; reporting the TAG pose as the object pose.",
                    obj_name.c_str());
            } else {
                t_tag_to_obj.setOrigin(offset_it->second);
            }

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

        if (!obj.observations.empty()) {
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