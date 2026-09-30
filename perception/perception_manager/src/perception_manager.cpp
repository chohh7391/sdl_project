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

    priority_cameras_ = this->declare_parameter<std::vector<std::string>>(
        "priority_cameras", std::vector<std::string>{"camera_3"});
    max_obs_age_s_ = this->declare_parameter<double>("max_observation_age_s", 0.5);
    // The newest image each camera has produced, from its camera_info (same
    // stamp as the image). A detection is current only if it is this recent in
    // its OWN camera's clock: tf2 keeps a tag's last transform indefinitely, so
    // a camera that stopped seeing a tag would otherwise keep contributing its
    // old view, stamped by fusion with the newest observation's time.
    for (const auto& cam : camera_frames_) {
        camera_info_subs_.push_back(this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/" + cam + "/camera_info", rclcpp::SensorDataQoS(),
            [this, cam](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
                rclcpp::Time t(msg->header.stamp);
                // The simulator's clock starts again from zero when it rebuilds
                // its world (the tool change every trial begins with). tf2 then
                // discards the new transforms as data from the past and keeps
                // answering with the old ones, so start over. clear() keeps the
                // static transforms (the wrist camera's mount).
                for (const auto& kv : latest_image_stamp_) {
                    if ((kv.second - t).seconds() > 1.0) {
                        RCLCPP_WARN(this->get_logger(),
                            "%s clock went back %.1f s -> %.1f s (simulator rebuilt); clearing TF",
                            cam.c_str(), kv.second.seconds(), t.seconds());
                        tf_buffer_->clear();
                        latest_image_stamp_.clear();
                        // ...and what was fused from the old transforms, or the
                        // publish timer sends it once more and puts the old
                        // stamps straight back into every listener's buffer.
                        for (auto& kv : objects_) {
                            kv.second.processed.clear();
                        }
                        break;
                    }
                }
                auto it = latest_image_stamp_.find(cam);
                if (it == latest_image_stamp_.end() || t > it->second) {
                    latest_image_stamp_[cam] = t;
                }
            }));
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
    // 0.18-0.20 m so neighbouring glassware cannot cover it; the vessel centre is
    // then BELOW the tag. beaker centre 0.0675 m, flask centre 0.0600 m.
    // The plate heights come from config/tag_mount.yaml through the launch
    // (the file task.py raises the plates with); SDL_TAG_Z sets both.
    double tag_z_beaker = this->declare_parameter<double>("tag_z_beaker", 0.18);
    double tag_z_flask = this->declare_parameter<double>("tag_z_flask", 0.20);
    if (std::getenv("SDL_TAG_Z")) {
        tag_z_beaker = tag_z_flask = std::atof(std::getenv("SDL_TAG_Z"));
    }
    RCLCPP_INFO(this->get_logger(), "tag plates at z beaker %.3f m, flask %.3f m",
                tag_z_beaker, tag_z_flask);
    tag_to_object_["beaker"] = tf2::Vector3(-0.15, 0.0, 0.0675 - tag_z_beaker);
    tag_to_object_["flask"] = tf2::Vector3(-0.15, 0.0, 0.0600 - tag_z_flask);

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

        // Collect one CURRENT observation per camera that can see the tag. Each
        // detector publishes its tags as <camera>_<tag> (apriltag.launch.py), so
        // this lookup is that camera's own detection and no other's.
        obj.observations.clear();
        for (const auto& cam : camera_frames_) {
            const std::string cam_tag = cam + "_" + tag_id;
            try {
                if (!tf_buffer_->canTransform("base_link", cam, tf2::TimePointZero) ||
                    !tf_buffer_->canTransform(cam, cam_tag, tf2::TimePointZero)) {
                    continue;
                }
                auto cam_to_tag = tf_buffer_->lookupTransform(cam, cam_tag, tf2::TimePointZero);
                auto newest = latest_image_stamp_.find(cam);
                if (newest == latest_image_stamp_.end()) {
                    continue;
                }
                const double age =
                    (newest->second - rclcpp::Time(cam_to_tag.header.stamp)).seconds();
                // Stamped AFTER this camera's newest image: seen before the
                // simulator rebuilt and restarted its clock, delivered after
                // the buffer was cleared for it (a detection still in flight).
                // tf2 keeps answering with it while its stamp stays the newest,
                // so the age test above would pass it as current: 20261001f
                // planned 34 of 90 perception trials from the scene before the
                // rebuild, one 77 mm off. Clear again and wait for new images.
                if (age < -kFutureTolS) {
                    RCLCPP_WARN(this->get_logger(),
                        "%s: %s stamped %.1f s after the newest image (from before the "
                        "simulator rebuilt); clearing TF", cam.c_str(), cam_tag.c_str(), -age);
                    tf_buffer_->clear();
                    obj.observations.clear();
                    break;
                }
                if (age > max_obs_age_s_) {
                    continue;
                }

                // base_link->camera AT THE INSTANT THE TAG WAS SEEN, not the
                // latest one. For the two fixed cell cameras the transform is
                // constant and the two are the same lookup. For a camera on
                // the wrist they are not: the recovery scan detects tags while
                // the arm is moving, so composing an older detection with the
                // arm's newest pose displaces the object by roughly (wrist
                // speed) x (detection latency). Falls back to the latest
                // transform when the buffer cannot answer at that stamp (a
                // static publisher outside the cache window).
                geometry_msgs::msg::TransformStamped base_to_cam;
                try {
                    base_to_cam = tf_buffer_->lookupTransform(
                        "base_link", cam, rclcpp::Time(cam_to_tag.header.stamp),
                        rclcpp::Duration::from_seconds(0.0));
                } catch (const tf2::TransformException &) {
                    base_to_cam = tf_buffer_->lookupTransform(
                        "base_link", cam, tf2::TimePointZero);
                    // The same for the camera's own pose (the arm's, for the
                    // wrist camera): a latest transform from before the rebuild
                    // is where the arm was then. A static mount is stamped 0.
                    if ((rclcpp::Time(base_to_cam.header.stamp) - newest->second).seconds() > kFutureTolS) {
                        RCLCPP_WARN(this->get_logger(),
                            "base_link -> %s is from before the simulator rebuilt; clearing TF",
                            cam.c_str());
                        tf_buffer_->clear();
                        obj.observations.clear();
                        break;
                    }
                }

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

        // Split by trust. A priority camera (the wrist camera) that sees the tag
        // right now REPLACES the fixed cameras in the object's own frame rather
        // than being blended with them: from the recovery scan it looks at the
        // tag from a fraction of the fixed cameras' range, which is the view to
        // believe. Both halves are also published on their own, so a consumer
        // can ask what the fixed cameras alone say (the pre-execution check).
        std::vector<CameraObservation> fixed, wrist;
        for (const auto& ob : obj.observations) {
            const bool prio = std::find(priority_cameras_.begin(), priority_cameras_.end(),
                                        ob.camera) != priority_cameras_.end();
            (prio ? wrist : fixed).push_back(ob);
        }
        obj.processed.clear();
        geometry_msgs::msg::TransformStamped out;
        if (fuse(obj_name, fixed, out)) {
            out.child_frame_id = obj_name + "_fixed";
            obj.processed["_fixed"] = out;
        }
        if (fuse(obj_name, wrist, out)) {
            out.child_frame_id = obj_name + "_wrist";
            obj.processed["_wrist"] = out;
        }
        if (fuse(obj_name, wrist.empty() ? fixed : wrist, out)) {
            out.child_frame_id = obj_name;
            obj.processed[""] = out;
        }
    }
}

bool PerceptionManager::fuse(const std::string& obj_name,
                             const std::vector<CameraObservation>& obs,
                             geometry_msgs::msg::TransformStamped& out) {
    if (obs.empty()) {
        return false;
    }
    geometry_msgs::msg::TransformStamped final_tf_msg;

    // 1. Tag 위치 융합 (Tag Pose Fusion)
    // Range-weighted over however many cameras saw the tag: a nearer view is a
    // better one, so each is weighted by 1/d^2. Rotation is blended by
    // successive slerp with the running weight, which for two cameras is
    // exactly q1.slerp(q2, w2/(w1+w2)).
    if (obs.size() == 1) {
        final_tf_msg = obs.front().pose_in_base;
    } else {
        const double eps = 1e-6;
        std::vector<double> w;
        double w_sum = 0.0;
        for (const auto& ob : obs) {
            double wi = 1.0 / (ob.distance * ob.distance + eps);
            w.push_back(wi);
            w_sum += wi;
        }

        // 시간: 가장 최신 관측 사용
        rclcpp::Time newest(obs.front().pose_in_base.header.stamp);
        for (const auto& ob : obs) {
            rclcpp::Time t(ob.pose_in_base.header.stamp);
            if (t > newest) { newest = t; }
        }
        final_tf_msg.header.stamp = newest;

        double x = 0.0, y = 0.0, z = 0.0;
        for (size_t i = 0; i < obs.size(); ++i) {
            const auto& tr = obs[i].pose_in_base.transform.translation;
            const double wi = w[i] / w_sum;
            x += tr.x * wi;
            y += tr.y * wi;
            z += tr.z * wi;
        }
        final_tf_msg.transform.translation.x = x;
        final_tf_msg.transform.translation.y = y;
        final_tf_msg.transform.translation.z = z;

        tf2::Quaternion q_fused;
        tf2::fromMsg(obs.front().pose_in_base.transform.rotation, q_fused);
        double w_run = w[0];
        for (size_t i = 1; i < obs.size(); ++i) {
            tf2::Quaternion qi;
            tf2::fromMsg(obs[i].pose_in_base.transform.rotation, qi);
            q_fused = q_fused.slerp(qi, w[i] / (w_run + w[i]));
            w_run += w[i];
        }
        final_tf_msg.transform.rotation = tf2::toMsg(q_fused);
    }

    // 2. 실제 Object 위치로 오프셋 적용 (Offset Application): tag -> object,
    // per object, because the plate sits above the vessel's centre by an
    // amount that differs between the two vessels (see tag_to_object_).
    tf2::Transform t_tag_fused;
    tf2::fromMsg(final_tf_msg.transform, t_tag_fused);
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
    final_tf_msg.transform = tf2::toMsg(t_tag_fused * t_tag_to_obj);
    final_tf_msg.header.frame_id = "base_link";
    out = final_tf_msg;
    return true;
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
        for (const auto& kv : objects_[obj_name].processed) {
            tf_broadcaster_->sendTransform(kv.second);
        }
    }
}

void PerceptionManager::publish_processed_ft() {
    ft_pub_->publish(ft_data_.processed_data);
}

void PerceptionManager::publish_processed_scale() {
    scale_pub_->publish(scale_data_.processed_data);
}